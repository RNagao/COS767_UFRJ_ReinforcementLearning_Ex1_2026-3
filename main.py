import resource
import os
import json

import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from cleaning_robot.env import CleaningEnv
from cleaning_robot.model import TabularModel

SEED = 42

def main(tasks=[1,2,3]):
    # task_1() if 1 in tasks else None
    # task_2() if 2 in tasks else None
    task_3() if 3 in tasks else None
    # analyze_task_1() if 1 in tasks else None
    # analyze_task_2() if 2 in tasks else None


def task_1():
    from cleaning_robot.policy import GreedyPolicy, RandomPolicy
    from cleaning_robot.dp import policy_evaluation, tabulate, tabulate_probs

    gammas = [0.7, 0.99]
    Ks = [0, 1, 2, 4, 6, 10]
    eval_modes = {
        'iterative': False,
        'direct': True,
    }

    path = "task_1.csv"
    done = set()
    if os.path.exists(path):
      prev = pd.read_csv(path)
      done = {(r.Policy, r.Evaluation_method, int(r.K), float(r.Gamma)) for r in prev.itertuples()}

    for k in Ks:
        env =  CleaningEnv(k=k)
        model = TabularModel(env)

        policies = {
            'gp': tabulate(model, GreedyPolicy(env)),
            'rp': tabulate_probs(model, RandomPolicy())
        }

        for gamma in gammas:
            for eval_mode, is_direct in eval_modes.items():
                for name, policy in policies.items():
                    if (name, eval_mode, k, gamma) in done:
                        continue

                    print(f"'Policy': {name}, 'Evaluation_method': {eval_mode}, 'K': {k}, 'Gamma': {gamma}")
                    t_i = time.perf_counter()
                    mem_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    try:
                        V = policy_evaluation(model, policy, gamma, method_direct=is_direct)
                    except Exception as e:
                        print(e)
                        V = None
                    mem_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    elapsed_time = time.perf_counter() - t_i

                    data = {
                        'Policy': name,
                        'Evaluation_method': eval_mode,
                        'K': k,
                        'Gamma': gamma,
                        'V_mean': np.mean(V) if V is not None else None,
                        'V_std': np.std(V) if V is not None else None,
                        'V_p10': np.percentile(V, 10) if V is not None else None,
                        'V_p90': np.percentile(V, 90) if V is not None else None,
                        'Mem_before': mem_before,
                        'Mem_after': mem_after,
                        'Elapsed_time': elapsed_time
                    }
                    pd.DataFrame([data]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)

        del model, policies
        gc.collect()

    return pd.read_csv(path)

def analyze_task_1():
    df = pd.read_csv("task_1.csv")

     # (a) Quanto uma política perde para a outra (V_mean)
    df_it = df[df['Evaluation_method'] == 'iterative']

    diffs = df_it.pivot_table(index=['K', 'Gamma'], columns='Policy',
                                values='V_mean', aggfunc='first')
    diffs.columns.name = None
    diffs = diffs.reset_index()

    diffs['loss_abs'] = diffs['gp'] - diffs['rp']
    diffs['loss_norm'] = diffs['loss_abs'] * (1 - diffs['Gamma'])
    diffs['loss_rel'] = diffs['loss_abs'] / diffs['rp'].abs().replace(0, np.nan)
    diffs['winner'] = np.select(
        [np.isclose(diffs['gp'], diffs['rp']), diffs['gp'] > diffs['rp']],
        ['draw', 'gp'],
        default='rp',
    )

    diffs.to_json('task1_policy.json', orient='records', indent=2, force_ascii=False)

    # c) Eval methods time and memory
    results = {}
    for method in ['iterative', 'direct']:
        df_m = df[df['Evaluation_method'] == method]

        failed = df_m[df_m['V_mean'].isna()]
        df_m = df_m.dropna(subset=['V_mean'])

        mem_delta = df_m['Mem_after'] - df_m['Mem_before']   # KB on Linux

        results[method] = {
            'memory_failures': int(len(failed)),
            'failed_cases': failed[['K', 'Gamma', 'Policy']].to_dict(orient='records'),
            'runs_ok': int(len(df_m)),  
            'mean_time_s': float(df_m['Elapsed_time'].mean()),
            'mean_mem_after_kb': float(df_m['Mem_after'].mean()),
            'mean_mem_delta_kb': float(mem_delta.mean()),
        }

    with open('task1_methods.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=float)

    return None


def task_2():
    from cleaning_robot.dp import PolicyIteration, ValueIteration

    os.makedirs('task_2_models', exist_ok=True)

    gammas = [0.7, 0.99]
    Ks = [0, 1, 2, 4, 6, 10]

    path = "task_2.csv"
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = {(r.Method, int(r.K), float(r.Gamma)) for r in prev.itertuples()}

    for k in Ks:
        env =  CleaningEnv(k=k)
        model = TabularModel(env)

        for gamma in gammas:
            methods = {
                'Value_iteration': ValueIteration,
                'Policy_iteration': PolicyIteration,
            }

            for name, method in methods.items():
                if (name, k, gamma) in done:
                    continue

                t_i = time.perf_counter()
                try:
                    result = method(model, gamma)
                except Exception as e:
                    print(e)
                    result = None
                elapsed_time = time.perf_counter() - t_i

                V = None
                if result is not None:
                    V = result.V
                    np.save(f'task_2_models/V_{name}_k{k}_gamma{gamma}.npy', result.V)
                    np.save(f'task_2_models/A_{name}_k{k}_gamma{gamma}.npy', result.actions)
            
                data = {
                    'Method': name,
                    'K': k,
                    'Gamma': gamma,
                    'Elapsed_time': elapsed_time,
                    'Iterations': result.count,
                    'V_mean': np.mean(V) if V is not None else None,
                    'V_std': np.std(V) if V is not None else None,
                    'V_p10': np.percentile(V, 10) if V is not None else None,
                    'V_p90': np.percentile(V, 90) if V is not None else None,
                }
                pd.DataFrame([data]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)


        del model, methods
        gc.collect()


    return pd.read_csv(path)

def analyze_task_2():
    gammas = [0.7, 0.99]
    Ks = [0, 1, 2, 4, 6, 10]
    
    # generate time and iteration by k graphs
    def bar_por_k(df, coluna, ylabel, titulo, arquivo):
      # média por K, uma série por (Policy, Evaluation_method)
      tab = (df.groupby(['K', 'Method', 'Gamma'])[coluna]
               .mean()
               .unstack(['Method', 'Gamma']))
      tab.columns = [f"{p} | {m}" for p, m in tab.columns]

      ax = tab.plot(kind='bar', figsize=(10, 5), width=0.8, edgecolor='white')
      ax.set_xlabel('K')
      ax.set_ylabel(ylabel)
      ax.set_title(titulo)
      ax.tick_params(axis='x', rotation=0)
      ax.set_yscale('log')
      ax.grid(axis='y', alpha=0.3)
      ax.set_axisbelow(True)
      ax.spines[['top', 'right']].set_visible(False)
      ax.legend(title='Política | Avaliação', frameon=False,
                bbox_to_anchor=(1.01, 1), loc='upper left')
      plt.tight_layout()
      plt.savefig(arquivo, dpi=150, bbox_inches='tight')
      plt.close()

    df = pd.read_csv("task_2.csv")

    bar_por_k(df, 'Elapsed_time', 'Tempo (s)', 'Tempo de execução × K',
    'tempo_x_k.png')
    bar_por_k(df, 'Iterations', 'Nº de iterações', 'Número de iterações × K', 
    'iter_x_k.png')

    # Calculate percentage of actions changed by state
    def load_actions(method, k, gamma):
        path = f'task_2_models/A_{method}_k{k}_gamma{gamma}.npy'
        if os.path.exists(path):
            return np.load(path)
        return None

    def pct_diff(a, b):
        if a is None or b is None:
            return np.nan
        if a.shape != b.shape:        
            return np.nan
        return (a != b).mean() * 100

    def matriz_mudancas(method):
        combos = (df[['K', 'Gamma']].drop_duplicates()
                    .sort_values(['Gamma', 'K'])
                    .itertuples(index=False))
        combos = [(int(k), g) for k, g in combos]
        labels = [f'k={k}, γ={g}' for k, g in combos]
        acts = [load_actions(method, k, g) for k, g in combos]

        n = len(combos)
        M = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                M[i, j] = pct_diff(acts[i], acts[j])
        return pd.DataFrame(M, index=labels, columns=labels)

    def plot_heatmap(mat, titulo, arquivo):
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(mat.values, cmap='viridis', vmin=0, vmax=100)
        ax.set_xticks(range(len(mat)), mat.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(mat)), mat.index)
        for i in range(len(mat)):
            for j in range(len(mat)):
                v = mat.values[i, j]
                txt = '—' if np.isnan(v) else f'{v:.1f}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=7,
                        color='white' if (np.isnan(v) or v < 50) else 'black')
        fig.colorbar(im, ax=ax, label='% de estados com ação diferente')
        ax.set_title(titulo)
        plt.tight_layout()
        plt.savefig(arquivo, dpi=150, bbox_inches='tight')
        plt.close()

    for method in ['Value_iteration', 'Policy_iteration']:
        mat = matriz_mudancas(method)
        mat.round(2).to_csv(f'task2_mudancas_{method}.csv')
        plot_heatmap(mat, f'Ações diferentes entre (K, γ) — {method}',
                    f'task2_mudancas_{method}.png')

    # Compare Vs with Item 1
    df_1 = pd.read_csv("task_1.csv")
    df1_V_mean = df_1.loc[df_1.groupby(['K', 'Gamma'])['V_mean'].idxmax()]
    df2_V_mean = df.loc[df.groupby(['K', 'Gamma'])['V_mean'].idxmax()]
    best_policies = {}
    for k in Ks:
        for gamma in gammas:
            best_df1_V_mean = df1_V_mean[(df1_V_mean['K'] == k) & (df1_V_mean['Gamma'] == gamma)]
            best_df2_V_mean = df2_V_mean[(df2_V_mean['K'] == k) & (df2_V_mean['Gamma'] == gamma)]
            best_policies[f'k{k}_gamma{gamma}'] = best_df1_V_mean['Policy'].item() if best_df1_V_mean['V_mean'].item() > best_df2_V_mean['V_mean'].item() else best_df2_V_mean['Method'].item()

    with open('task2_best_policies.json', 'w', encoding='utf-8') as f:
      json.dump(best_policies, f, indent=2, ensure_ascii=False)

    return None

def task_3():
    from cleaning_robot.qlearning import QLearning
    from cleaning_robot.dp import policy_evaluation

    scenarios = ((1, 0.99), (6, 0.99))
    seeds = [0, 10, 42]
    lrs = [0.1]
    epsilons = [0.1]
    n_steps = 1e8

    path = "task_3.csv"
    done = set()
    if os.path.exists(path):
        prev = pd.read_csv(path)
        done = {(int(r.Seed), int(r.N_steps), int(r.K), float(r.Gamma), float(r.Q_init), float(r.Lr), float(r.Epsilon)) for r in prev.itertuples()}

    for k, gamma in scenarios:
        q_inits = [0.0, -1/(1-gamma), -1/(2 * (1 - gamma))]
        env =  CleaningEnv(k=k)
        for seed in seeds:
            for q_init in q_inits:
                for lr in lrs:
                    for epsilon in epsilons:
                        print(f"QL_k{k}_gamma{gamma}_seed{seed}_qinit{q_init}_lr{lr}_epsilon{epsilon}")
                        if (seed, n_steps, k, gamma, q_init, lr, epsilon) in done:
                            continue
                        
                        try:
                            t_i = time.perf_counter()
                            ql = QLearning(env, gamma, n_steps, lr = lr, epsilon = epsilon, seed=seed, q_init=q_init)
                            elapsed_time = time.perf_counter() - t_i
                        except Exception as e:
                            elapsed_time = time.perf_counter() - t_i
                            print(e)
                            ql = None

                        V = None
                        model = None
                        if ql is not None:
                            policy = ql.actions
                            model = TabularModel(env)
                            try:
                                V = policy_evaluation(model, policy, gamma, method_direct=False)
                            except Exception as e:
                                print(e)

                        data = {
                            'Seed': seed,
                            'K': k,
                            'Gamma': gamma,
                            'Q_init': q_init,
                            'Lr': lr,
                            'Epsilon': epsilon,
                            'Elapsed_time': elapsed_time,
                            'N_steps': n_steps,
                            'V_mean': np.mean(V) if V is not None else None,
                            'V_std': np.std(V) if V is not None else None,
                            'V_p10': np.percentile(V, 10) if V is not None else None,
                            'V_p90': np.percentile(V, 90) if V is not None else None,
                        }
                        pd.DataFrame([data]).to_csv(path, mode='a', header=not os.path.exists(path), index=False)

        # if model is not None:
        #     del model
        # gc.collect()

    return pd.read_csv(path)


if __name__ == '__main__':
    main(tasks=[3])