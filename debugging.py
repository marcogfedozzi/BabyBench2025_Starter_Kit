
import os
import yaml
import math
import torch
from babybench import rl_utils as rlu
import evaluation
import torchrl.modules as trm

rlu.register_script_resolvers()
run_id = '4z8l5fit' #'4wr9eh6t'
run_dir = os.path.join('models', f'run_{run_id}')
cfg = evaluation._get_hydra_config(run_dir)

max_grad_checks = 20

with open('examples/config_selftouch_simple.yml') as f:
    bcfg = yaml.safe_load(f)
bcfg = rlu.update_config_savedir(bcfg, run_id)
env = rlu.make_env(cfg, bcfg, is_eval=True)

print(env.observation_spec)
print(env.action_spec)


def find_normal_extractors(module):
    from torchrl.modules import NormalParamExtractor
    found = []
    for name, m in module.named_modules():
        if isinstance(m, NormalParamExtractor):
            found.append((name, m))
    return found


def inspect_extractor(name, extr):
    print(f"\nExtractor: {name}  class={type(extr)}")
    # print basic attrs
    attrs = [a for a in dir(extr) if not a.startswith('_')]
    print('public attrs (sample):', attrs[:20])
    # parameters
    print('named_parameters:')
    for pn, p in extr.named_parameters():
        print(' ', pn, tuple(p.shape), 'mean,std=', float(p.mean()), float(p.std()))
    # buffers
    print('named_buffers:')
    for bn, b in extr.named_buffers():
        print(' ', bn, tuple(b.shape), 'mean,std=', float(b.mean()), float(b.std()))


def compute_hidden_to_loc_jacobian(agent, td0, max_outputs=32):
    """Compute jacobian of loc w.r.t. hidden for a single batch element using autograd.
    Returns stats dictionary. If autograd path fails, returns None.
    """
    # ensure gradients are enabled
    torch.set_grad_enabled(True)
    agent.train()  # enable grad tracking in modules
    out = agent(td0)
    # expect tensordict-like outputs
    try:
        hidden = out.get('hidden')
        loc = out.get('loc')
    except Exception:
        print('Could not retrieve hidden/loc from agent output; keys:', list(out.keys()))
        return None

    # assume batch-first; take first element
    if hidden.ndim == 2 and loc.ndim == 2:
        h = hidden[0]
        l = loc[0]
    else:
        # fallback: flatten batch into single vector
        h = hidden.reshape(-1)
        l = loc.reshape(-1)

    if not h.requires_grad:
        # attempt to enable grad on the tensor in-place (works if it's a view)
        try:
            h.requires_grad_(True)
        except Exception:
            print('hidden does not require grad; autograd jacobian will likely fail')

    h_shape = tuple(h.shape)
    l_shape = tuple(l.shape)
    print(f'Computing jacobian: hidden shape={h_shape}, loc shape={l_shape}')

    # limit output dims
    out_dim = l.numel()
    in_dim = h.numel()
    nout = min(out_dim, max_outputs)

    jac_rows = []
    for i in range(nout):
        # pick scalar output l_flat[i]
        li = l.reshape(-1)[i]
        grad = torch.autograd.grad(li, h, retain_graph=True, allow_unused=True)[0]
        if grad is None:
            jac_rows.append(torch.zeros_like(h))
        else:
            jac_rows.append(grad.detach().cpu())

    J = torch.stack(jac_rows, dim=0)  # (nout, hidden_dim)
    stats = {
        'J_shape': tuple(J.shape),
        'J_fro_norm': float(torch.norm(J).item()),
        'J_row_maxabs': float(J.abs().max().item()),
        'J_row_meanabs': float(J.abs().mean().item()),
    }
    return stats


def run_for_agent(agent, tag):
    print('\n==== RUN:', tag, '====')
    # load an env observation
    td0 = env.reset()

    # move agent to cpu
    agent = agent.to('cpu')
    agent.eval()

    # inspect extractors
    extrs = find_normal_extractors(agent)
    if not extrs:
        print('No NormalParamExtractor instances found in agent')
    else:
        for name, extr in extrs:
            inspect_extractor(name, extr)

    # --- Additional instrumentation: capture extractor inputs and upstream linear outputs ---
    hooks = []
    hooks_data = {'linear_outputs': {}, 'extractor_inputs': {}}

    from tensordict.nn.distributions.continuous import NormalParamExtractor

    for name, m in agent.named_modules():
        if 'module.1' in name:
            # capture linear outputs under policy path
            if isinstance(m, torch.nn.Linear):
                def make_lin_hook(n):
                    def _h(mod, inp, out):
                        try:
                            hooks_data['linear_outputs'][n] = out.detach().cpu()
                        except Exception:
                            hooks_data['linear_outputs'][n] = None
                    return _h

                hooks.append(m.register_forward_hook(make_lin_hook(name)))

            # capture extractor inputs
            if isinstance(m, NormalParamExtractor):
                def make_ex_hook(n):
                    def _h(mod, inp, out):
                        # extractor usually receives a single tensor as first arg
                        if len(inp):
                            hooks_data['extractor_inputs'][n] = inp[0]
                        else:
                            hooks_data['extractor_inputs'][n] = None
                    return _h

                hooks.append(m.register_forward_hook(make_ex_hook(name)))

    # forward once to populate hooks_data
    with torch.no_grad():
        out = agent(td0)

    # remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # report hidden / extractor inputs
    try:
        hidden = out.get('hidden')
        print('\nhidden requires_grad =', hidden.requires_grad, ' grad_fn=', type(hidden.grad_fn).__name__ if hidden.grad_fn is not None else None)
        print('hidden mean,std =', float(hidden.mean()), float(hidden.std()))
    except Exception:
        hidden = None
        print('Could not read hidden from agent output')

    for ename, et in hooks_data['extractor_inputs'].items():
        if et is None:
            print(f'extractor {ename}: input captured = None')
            continue
        print(f'\nextractor {ename}: input shape={tuple(et.shape)} requires_grad={et.requires_grad} grad_fn={type(et.grad_fn).__name__ if et.grad_fn is not None else None}')
        try:
            print('  mean,std =', float(et.mean()), float(et.std()))
        except Exception:
            pass
        # compare to hidden if shapes compatible
        if hidden is not None:
            try:
                # try to align last dims: if et flattened equals hidden flattened, compare norms
                h_flat = hidden.reshape(-1).detach().cpu()
                et_flat = et.reshape(-1).detach().cpu()
                if h_flat.numel() == et_flat.numel():
                    diff = (h_flat - et_flat).norm().item()
                    print('  norm(hidden - extractor_input) =', diff)
                else:
                    print('  shapes differ: hidden', tuple(hidden.shape), 'extractor_in', tuple(et.shape))
            except Exception as e:
                print('  comparison failed:', e)

    # print nearby linear outputs summary
    if hooks_data['linear_outputs']:
        print('\ncaptured linear outputs:')
        for lname, t in hooks_data['linear_outputs'].items():
            if t is None:
                print(' ', lname, '-> None')
                continue
            try:
                print(' ', lname, 'shape=', tuple(t.shape), 'mean,std=', float(t.mean()), float(t.std()))
            except Exception:
                print(' ', lname, 'shape=', tuple(t.shape))

    # compute jacobian hidden -> loc (try again but this time run forward with grad enabled)
    jac_stats = compute_hidden_to_loc_jacobian(agent, td0)
    print('\njacobian stats (eval-forward):', jac_stats)

    # --- Patch 1: live jacobian by enabling grad on inputs and forwarding with grad ---
    def enable_td_requires_grad(td):
        # set all floating tensors in the tensordict to require grad
        try:
            keys = list(td.keys())
        except Exception:
            # tensordict older API
            keys = []
            for k in td._key_iter():
                keys.append(k)
        tdg = td.clone()
        for k in keys:
            try:
                v = tdg.get(k)
            except Exception:
                continue
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                tdg.set(k, v.detach().clone().requires_grad_(True))
        return tdg

    print('\n-- Live jacobian test: enabling grad on inputs and forwarding --')
    try:
        agent.train()
        tdg = enable_td_requires_grad(td0)
        jac_live = compute_hidden_to_loc_jacobian(agent, tdg)
        print('jacobian stats (live):', jac_live)
    except Exception as e:
        print('live jacobian test failed:', e)

    # --- Patch 2: clamp learned scale and sample to see if actions become loc-driven ---
    from torchrl.modules import TanhNormal

    def clamp_scale_and_sample(agent, td, factor=0.01, samples=1000):
        # run in eval no grad
        agent = agent.to('cpu')
        agent.eval()
        with torch.no_grad():
            out = agent(td)
            loc = out.get('loc')
            scale = out.get('scale')
        print('\nOriginal loc mean,std =', float(loc.mean()), float(loc.std()))
        print('Original scale mean,std =', float(scale.mean()), float(scale.std()))
        # clamp scale
        s_clamped = scale * factor
        try:
            # sample many times
            a_samples = TanhNormal(loc, s_clamped).rsample((samples,))
            # a_samples shape: (samples, *action_shape)
            a_mean = a_samples.mean(dim=0)
            a_std = a_samples.std(dim=0)
            print(f'Clamped scale factor={factor}: sampled action mean (per-dim) mean,std =', float(a_mean.mean()), float(a_std.mean()))
        except Exception as e:
            print('sampling under clamped scale failed:', e)

    clamp_scale_and_sample(agent, td0, factor=0.01, samples=1000)


def run_detach_probe(agent, env, key='touch', max_grad_checks=8):
    """Probe where the autograd graph detaches for input tensordict key.

    - creates a requires_grad copy of td[key]
    - forwards td through the agent's common_operator (the module that consumes `key`)
    - attaches forward hooks to inner submodules and records outputs
    - for each captured output: prints requires_grad/grad_fn and attempts a small autograd test
    """
    print('\n==== DETACH PROBE for key:', key, '====')
    td0 = env.reset()
    try:
        x = td0.get(key)
    except Exception:
        x = td0[key]
    print('orig', key, 'shape', tuple(x.shape), 'dtype', x.dtype, 'requires_grad', x.requires_grad)

    xg = x.detach().clone().requires_grad_(True)
    # clone td and set key
    try:
        tdg = td0.clone()
    except Exception:
        import copy
        tdg = copy.deepcopy(td0)
    try:
        tdg.set(key, xg)
    except Exception:
        tdg[key] = xg

    # find common operator by in_keys
    common = None
    common_name = None
    for name, m in agent.named_modules():
        try:
            ik = getattr(m, 'in_keys', None)
            if ik is None:
                continue
            if isinstance(ik, (list, tuple)):
                if key in ik:
                    common = m
                    common_name = name
                    break
            else:
                if ik == key:
                    common = m
                    common_name = name
                    break
        except Exception:
            continue
    if common is None:
        print('Could not find common operator that consumes key', key)
        return
    print('found common operator at', common_name, 'type', type(common))

    inner = getattr(common, 'module', None)
    if inner is None:
        print('common has no .module attribute; will probe common directly')
        inner = common

    # attach hooks to all inner submodules
    captured = {}
    hooks = []
    for name, m in inner.named_modules():
        def mk(n):
            def h(mod, inp, out):
                # store out tensor (or first tensor if tuple)
                try:
                    if isinstance(out, torch.Tensor):
                        captured.setdefault(n, {})['out'] = out
                    elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                        captured.setdefault(n, {})['out'] = out[0]
                except Exception:
                    captured.setdefault(n, {})['out'] = None
            return h
        hooks.append(m.register_forward_hook(mk(name)))

    # forward with grad enabled
    with torch.enable_grad():
        try:
            outc = common(tdg)
        except Exception as e:
            print('forward of common failed:', e)
            # remove hooks
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            return

    # remove hooks
    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    # iterate captured modules in order and run grad test
    names = list(captured.keys())
    if not names:
        print('no inner module outputs captured')
        return

    print('\nCaptured module outputs:')
    for i, n in enumerate(names):
        info = captured[n]
        t = info.get('out', None)
        if t is None:
            print(f'{i:03d} {n}: output None')
            continue
        # move to cpu for printing
        tt = t.detach().cpu()
        req = t.requires_grad
        gfn = type(t.grad_fn).__name__ if t.grad_fn is not None else None
        print(f'{i:03d} {n}: shape={tuple(tt.shape)} dtype={tt.dtype} requires_grad={req} grad_fn={gfn} mean,std={float(tt.mean()):.6g},{float(tt.std()):.6g}')

        # attempt a small autograd test: grad of a scalar from this tensor w.r.t. xg
        try:
            scalar = t.reshape(-1)[0]
            grad = torch.autograd.grad(scalar, xg, retain_graph=True, allow_unused=True)[0]
            if grad is None:
                print('    grad wrt input: None')
            else:
                print('    grad wrt input: present, norm=', float(grad.detach().cpu().norm()))
        except Exception as e:
            print('    grad test failed:', e)

        # limit checks
        if i >= max_grad_checks:
            print('...skipping further grad checks')
            break

    # also try jacobian loc <- xg via full forward through policy if available
    try:
        # get hidden from outc if present
        hidden = outc.get('hidden') if hasattr(outc, 'get') else outc['hidden']
        print('\nHidden produced shape', tuple(hidden.shape), 'requires_grad', hidden.requires_grad)
        # find policy mlp and extractor as before
        from tensordict.nn.distributions.continuous import NormalParamExtractor
        extractor_name = None
        mlp_name = None
        for name, m in agent.named_modules():
            if isinstance(m, NormalParamExtractor):
                extractor_name = name
                parent = '.'.join(name.split('.')[:-1])
                mlp_name = parent + '.0'
                break
        if extractor_name is not None:
            mlp = agent.get_submodule(mlp_name)
            extractor = agent.get_submodule(extractor_name)
            mlp_out = mlp(hidden)
            loc, scale, *rest = extractor(mlp_out)
            print('loc shape', tuple(loc.shape), 'requires_grad', loc.requires_grad)
            # jacobian small set
            nout = min(4, loc.numel())
            J = []
            for i in range(nout):
                li = loc.reshape(-1)[i]
                grad = torch.autograd.grad(li, xg, retain_graph=True, allow_unused=True)[0]
                J.append(grad.detach().cpu() if grad is not None else torch.zeros_like(xg.detach().cpu()))
            J = torch.stack(J, dim=0)
            print('jacobian loc<-input shape', tuple(J.shape), 'fro_norm', float(torch.norm(J)))
        else:
            print('could not find extractor to run jacobian test')
    except Exception as e:
        print('jacobian loc<-input test failed:', e)


def run_vanish_pinpoint(agent, env, key='touch', threshold=1e-8, lookback=2):
    """Pinpoint the first submodule output whose gradient w.r.t. the input falls below `threshold`.

    Prints each captured module's grad-norm and reports the first module where gradient norm <= threshold.
    """
    print('\n==== VANISH PINPOINT for key:', key, 'threshold=', threshold, '====')
    td0 = env.reset()
    try:
        x = td0.get(key)
    except Exception:
        x = td0[key]

    xg = x.detach().clone().requires_grad_(True)
    try:
        tdg = td0.clone()
    except Exception:
        import copy
        tdg = copy.deepcopy(td0)
    try:
        tdg.set(key, xg)
    except Exception:
        tdg[key] = xg

    # find common operator consuming this key
    common = None
    for name, m in agent.named_modules():
        ik = getattr(m, 'in_keys', None)
        if ik is None:
            continue
        if isinstance(ik, (list, tuple)):
            if key in ik:
                common = m
                common_name = name
                break
        else:
            if ik == key:
                common = m
                common_name = name
                break
    if common is None:
        print('Could not find common operator that consumes key', key)
        return None
    inner = getattr(common, 'module', common)

    # attach hooks in order to capture outputs
    captured = []
    hooks = []
    for name, m in inner.named_modules():
        def mk(n):
            def h(mod, inp, out):
                try:
                    if isinstance(out, torch.Tensor):
                        captured.append((n, out))
                    elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                        captured.append((n, out[0]))
                    else:
                        captured.append((n, None))
                except Exception:
                    captured.append((n, None))
            return h
        hooks.append(m.register_forward_hook(mk(name)))

    with torch.enable_grad():
        try:
            _ = common(tdg)
        except Exception as e:
            print('forward failed during vanish pinpoint:', e)
            for h in hooks:
                try:
                    h.remove()
                except Exception:
                    pass
            return None

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    print(f'Captured {len(captured)} submodule outputs; computing gradient norms w.r.t. input...')
    vanish_idx = None
    norms = []
    for i, (n, t) in enumerate(captured):
        if t is None:
            print(f'{i:03d} {n}: output=None')
            norms.append(0.0)
            continue
        try:
            scalar = t.reshape(-1).abs().sum()
            grad = torch.autograd.grad(scalar, xg, retain_graph=True, allow_unused=True)[0]
            gnorm = 0.0 if grad is None else float(grad.detach().cpu().norm())
        except Exception as e:
            print(f'{i:03d} {n}: grad computation failed: {e}')
            gnorm = 0.0
        norms.append(gnorm)
        print(f'{i:03d} {n}: shape={tuple(t.shape)} grad_norm={gnorm:.6g}')
        if vanish_idx is None and gnorm <= threshold:
            vanish_idx = i
            print('\n>> Gradient vanishes at index', i, 'module', n, 'grad_norm=', gnorm)
            start = max(0, i - lookback)
            end = min(len(captured) - 1, i + lookback)
            print(' surrounding modules:')
            for j in range(start, end + 1):
                nj, tj = captured[j]
                print(f'  {j:03d} {nj}: shape={(None if tj is None else tuple(tj.shape))} grad_norm={norms[j]:.6g}')
            break

    if vanish_idx is None:
        print('\nNo vanishing point found (all grad norms > threshold)')
    return vanish_idx


def run_inspect_module(agent, env, key='touch', threshold=1e-8, eps_list=(1e-4, 1e-3, 1e-2)):
    """Find the vanish index and inspect that module's details and sensitivity.

    - runs `run_vanish_pinpoint` to locate the index
    - re-runs a forward capturing module inputs/outputs
    - prints module class, named parameters stats, buffer stats
    - runs finite-difference on the captured module input to estimate sensitivity
    """
    print('\n==== INSPECT MODULE NEAR VANISH POINT ===')
    vanish_idx = run_vanish_pinpoint(agent, env, key=key, threshold=threshold)
    if vanish_idx is None:
        print('No vanish index found; aborting module inspect')
        return

    # find common and inner module
    common = None
    for name, m in agent.named_modules():
        ik = getattr(m, 'in_keys', None)
        if ik is None:
            continue
        if isinstance(ik, (list, tuple)):
            if key in ik:
                common = m
                common_name = name
                break
        else:
            if ik == key:
                common = m
                common_name = name
                break
    if common is None:
        print('Could not find common operator')
        return
    inner = getattr(common, 'module', common)

    # capture inputs and outputs (pre and post) for each inner submodule
    captures = []
    hooks = []
    for name, m in inner.named_modules():
        def mk(n):
            def h(mod, inp, out):
                # store first tensor of inp if available and out
                in_t = None
                try:
                    if isinstance(inp, (list, tuple)) and len(inp) and isinstance(inp[0], torch.Tensor):
                        in_t = inp[0].detach().cpu()
                    elif isinstance(inp, torch.Tensor):
                        in_t = inp.detach().cpu()
                except Exception:
                    in_t = None
                out_t = None
                try:
                    if isinstance(out, torch.Tensor):
                        out_t = out.detach().cpu()
                    elif isinstance(out, (list, tuple)) and len(out) and isinstance(out[0], torch.Tensor):
                        out_t = out[0].detach().cpu()
                except Exception:
                    out_t = None
                captures.append((n, in_t, out_t))
            return h
        hooks.append(m.register_forward_hook(mk(name)))

    # run forward (no grad needed) to populate captures
    with torch.no_grad():
        _ = common(env.reset())

    for h in hooks:
        try:
            h.remove()
        except Exception:
            pass

    if vanish_idx >= len(captures):
        print('vanish_idx', vanish_idx, 'out of captured range', len(captures))
        return

    name, in_t, out_t = captures[vanish_idx]
    print('\nModule at vanish index', vanish_idx, '->', name)
    # get module object
    try:
        module_obj = inner.get_submodule(name)
    except Exception:
        # fallback: locate by iterating
        module_obj = None
        for nm, m in inner.named_modules():
            if nm == name:
                module_obj = m
                break
    if module_obj is None:
        print('Could not resolve module object for', name)
        return

    print('module class:', type(module_obj))
    # parameters
    print('\nParameters:')
    total_params = 0
    for pn, p in module_obj.named_parameters():
        ps = tuple(p.shape)
        total_params += p.numel()
        p_cpu = p.detach().cpu()
        print(' ', pn, ps, 'mean,std,min,max=', float(p_cpu.mean()), float(p_cpu.std()), float(p_cpu.min()), float(p_cpu.max()))
    print(' total params:', total_params)
    # buffers
    print('\nBuffers:')
    for bn, b in module_obj.named_buffers():
        b_cpu = b.detach().cpu()
        print(' ', bn, tuple(b_cpu.shape), 'mean,std=', float(b_cpu.mean()), float(b_cpu.std()))

    # print input/output summaries
    print('\nCaptured input shape, dtype:', None if in_t is None else (tuple(in_t.shape), in_t.dtype))
    print('Captured output shape, dtype:', None if out_t is None else (tuple(out_t.shape), out_t.dtype))
    if in_t is None:
        print('No captured input tensor for this module; cannot run sensitivity test')
        return

    # convert input to tensor on cpu with float
    inp = in_t.clone().float()

    # run finite-difference sensitivity tests on module_obj using the captured input
    print('\nFinite-difference sensitivity tests (approx output change / eps):')
    # ensure module on cpu and in eval
    module_obj = module_obj.to('cpu')
    module_obj.eval()
    try:
        with torch.no_grad():
            out0 = module_obj(inp.clone())
            if isinstance(out0, torch.Tensor):
                out0 = out0.detach()
            else:
                out0 = out0[0].detach()
    except Exception as e:
        print('module forward failed on captured input:', e)
        return

    for eps in eps_list:
        noise = torch.randn_like(inp) * float(eps)
        try:
            with torch.no_grad():
                out1 = module_obj((inp + noise))
                if isinstance(out1, torch.Tensor):
                    out1 = out1.detach()
                else:
                    out1 = out1[0].detach()
            diff = (out1 - out0).norm().item()
            print(f' eps={eps:.0e}  |out1-out0|_2 = {diff:.6g}  (norm/eps = {diff / eps:.6g})')
        except Exception as e:
            print(' eps test failed for', eps, ':', e)

    print('\nDone inspecting module', name)


if __name__ == '__main__':
    # prepare trained agent
    trained_agent = rlu.make_agent(cfg, env)
    # load without map_location then move to cpu per your instruction
    trained_sd = torch.load(os.path.join(run_dir, 'actor_module.pth'))
    trained_agent.load_state_dict(trained_sd)
    run_for_agent(trained_agent, 'trained')
    run_detach_probe(trained_agent, env, key='touch', max_grad_checks=max_grad_checks)

    # prepare fresh/untrained agent
    untrained_agent = rlu.make_agent(cfg, env)
    run_for_agent(untrained_agent, 'untrained')
    run_detach_probe(untrained_agent, env, key='touch', max_grad_checks=max_grad_checks)

    print('\nall done')
