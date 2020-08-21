from run_experiment_common import *

def visualize():
    search_pattern = 'rq_*_stats.p'
    filename = 'rq'

    iteration_results = glob.glob(os.path.join(DATA_DIR, search_pattern))
    aggregated_results = os.path.join(DATA_DIR, filename)
    df = stats.load_stats_dataframe(iteration_results, aggregated_results)
    df.loc[pd.isnull(df['rewardfun']), 'rewardfun'] = 'none'
    mean_df = df[(df['detected'] + df['missed']) > 0].groupby(['step', 'env', 'agent', 'rewardfun'],
                                                              as_index=False).mean()

    # Single figures
    #

    for (agent, rewfun) in [('mlpclassifier', 'APHFW_reward'), ('tableau', 'timerank')]:
        for (agent, rewfun) in [('mlpclassifier', 'APHFW_reward')]:
            i = 0

            for env in sorted(mean_df['env'].unique(), reverse=True):
                plotname = 'rq2_napfd_abs_%s_%s_%s' % (env, agent, rewfun)
                print(env)

                reddf = mean_df[(mean_df['env'] == env) & (
                    mean_df['agent'].isin([agent, 'heur_sort', 'heur_weight', 'heur_random'])) & (
                                    mean_df['rewardfun'].isin([rewfun, 'none']))].groupby(['step', 'agent']).mean()[
                    'napfd'].unstack()
                # reddf['Heuristic Sort'] = reddf['heur_sort'] - reddf[agent]
                # reddf['Heuristic Weight'] = reddf['heur_weight'] - reddf[agent]
                # reddf['Random'] = reddf['heur_random'] - reddf[agent]
                # reddf['Random'] = reddf['heur_random']
                # reddf['Heuristic Sort'] = reddf['heur_sort']
                reddf['APHFW_reward'] = reddf[agent]
                del reddf[agent]
                del reddf['heur_sort']
                del reddf['heur_weight']
                del reddf['heur_random']
                window_size = 25
                r = reddf.groupby(reddf.index // window_size).mean()
                xdf = r.stack()
                xdf = xdf.reset_index(level=xdf.index.names)
                xdf['step'] *= window_size

                fig = plt.figure(figsize=(8, 4))
                ax = sns.barplot(data=xdf, x='step', y=0, hue='agent', figure=fig)

                ax.set_ylabel('')
                ax.set_xlabel('CI Cycle')
                ax.legend_.remove()

                if i == 0:
                    ax.set_ylabel('NAPFD')
                    # ax.set_ylim([-0.6, 0.6])
                    ax.set_title('Paint Control')
                    ax.legend(ncol=1, loc=1)
                elif i == 1:
                    ax.set_ylabel('NAPFD')
                    ax.set_title('IOF/ROL')
                    ax.legend(ncol=1, loc=1)
                elif i == 2 and len(mean_df['env'].unique()) == 3:
                    ax.set_ylabel('NAPFD')
                    ax.set_title('Google GSDTSR')
                    ax.legend(ncol=1, loc=1)

                ax.set_ylim([0, 1])
                # ax.set_ylim([-0.6, 0.6])
                fig.tight_layout()
                save_figures(fig, plotname)
                plt.clf()

                i += 1

run_experiments(exp_run_industrial_datasets, parallel=PARALLEL)

visualize()

