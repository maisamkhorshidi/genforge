import numpy as np

def gp_track_param_ensemble(gp):
    
    gen = gp.state['generation']
    num_pop = gp.config['runcontrol']['num_pop']
    pop_size = gp.config['runcontrol']['pop_size']
    
    if gen == 0:
        # Initialize tracking parameters
        # Track
        gp.track = {
            'generation': [0],
            'population': [None],
            'complexity':{
                'isolated': [np.full((pop_size, num_pop), np.inf)],
                'ensemble':[np.full((pop_size), np.inf)],
                },
            'depth': {
                'isolated': [[[None for _ in range(pop_size)] for _ in range(num_pop)]],
                'ensemble': [[None for _ in range(pop_size)]],
                },
            'num_nodes':{
                'isolated': [[[None for _ in range(pop_size)] for _ in range(num_pop)]],
                'ensemble': [[None for _ in range(pop_size)]],
                },
            'weight_genes': [[[None for _ in range(pop_size)] for _ in range(num_pop)]],
            'ensemble_weight': [np.full((pop_size, num_pop), np.inf)],
            'fitness': {
                'isolated': {
                    'train': [np.full((pop_size, num_pop), np.inf)],
                    'validation': [np.full((pop_size, num_pop), np.inf)],
                    'test': [np.full((pop_size, num_pop), np.inf)],
                    },
                'ensemble': {
                    'train': [np.full((pop_size), np.inf)],
                    'validation': [np.full((pop_size), np.inf)],
                    'test': [np.full((pop_size), np.inf)],
                    },
                },
            'std_fitness': {
                'isolated': {
                    'train': [np.full((num_pop), np.inf)],
                    'validation': [np.full((num_pop), np.inf)],
                    'test': [np.full((num_pop), np.inf)],
                    },
                'ensemble': {
                    'train': [0],
                    'validation': [0],
                    'test': [0],
                    },
                },
            'mean_fitness': {
                'isolated': {
                    'train': [np.full((num_pop), np.inf)],
                    'validation': [np.full((num_pop), np.inf)],
                    'test': [np.full((num_pop), np.inf)],
                    },
                'ensemble': {
                    'train': [0],
                    'validation': [0],
                    'test': [0],
                    },
                },
            'idx_minus': {
                'mutation': [np.full((pop_size, num_pop), np.inf)],
                'crossover': [[np.full((pop_size, 2), np.inf) for _ in range(num_pop)]],
                'crossover_hi': [[np.full((pop_size, 2), np.inf) for _ in range(num_pop)]],
                'reproduction': [np.full((pop_size, num_pop), np.inf)],
                'elite_isolated': [np.full((pop_size, num_pop), np.inf)],
                'elite_ensemble': [np.full((pop_size, num_pop), np.inf)],
                },
            'ensemble_idx': [np.full((pop_size, num_pop), np.inf)],
            'rank': {
                'fitness':{
                    'isolated': {
                        'train': [[np.full((pop_size), np.inf) for _ in range(num_pop)]],
                        'validation': [[np.full((pop_size), np.inf) for _ in range(num_pop)]],
                        'test': [[np.full((pop_size), np.inf) for _ in range(num_pop)]],
                        },
                    'ensemble': {
                        'train': [np.full((pop_size), np.inf)],
                        'validation': [np.full((pop_size), np.inf)],
                        'test': [np.full((pop_size), np.inf)],
                        },
                    },
                'complexity':{
                    'isolated': [np.full((pop_size, num_pop), np.inf)],
                    'ensemble': [np.full((pop_size), np.inf)],
                    },
                },
            'all_ensemble':{
                'idx': [None],
                'fitness': {
                    'train': [None],
                    'validation': [None],
                    'test': [None],
                    },
                },
            }
        
    else:
        # Append tracking parameters
        gp.track['generation'].append(gen)
        gp.track['population'].append(None)
        gp.track['complexity']['isolated'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['complexity']['ensemble'].append(np.full((pop_size), np.inf))
        gp.track['depth']['isolated'].append([[None for _ in range(pop_size)] for _ in range(num_pop)])
        gp.track['depth']['ensemble'].append([None for _ in range(pop_size)])
        gp.track['num_nodes']['isolated'].append([[None for _ in range(pop_size)] for _ in range(num_pop)])
        gp.track['num_nodes']['ensemble'].append([None for _ in range(pop_size)])
        gp.track['weight_genes'].append([[None for _ in range(pop_size)] for _ in range(num_pop)])
        gp.track['ensemble_weight'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['fitness']['isolated']['train'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['fitness']['isolated']['validation'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['fitness']['isolated']['test'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['fitness']['ensemble']['train'].append([np.full((pop_size), np.inf)]),
        gp.track['fitness']['ensemble']['validation'].append(np.full((pop_size), np.inf))
        gp.track['fitness']['ensemble']['test'].append(np.full((pop_size), np.inf)),
        gp.track['std_fitness']['isolated']['train'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['std_fitness']['isolated']['validation'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['std_fitness']['isolated']['test'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['std_fitness']['ensemble']['train'].append(np.full((pop_size), np.inf))
        gp.track['std_fitness']['ensemble']['validation'].append(np.full((pop_size), np.inf))
        gp.track['std_fitness']['ensemble']['test'].append(np.full((pop_size), np.inf))
        gp.track['mean_fitness']['isolated']['train'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['mean_fitness']['isolated']['validation'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['mean_fitness']['isolated']['test'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['mean_fitness']['ensemble']['train'].append(np.full((pop_size), np.inf))
        gp.track['mean_fitness']['ensemble']['validation'].append(np.full((pop_size), np.inf))
        gp.track['mean_fitness']['ensemble']['test'].append(np.full((pop_size), np.inf))
        gp.track['idx_minus']['mutation'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['idx_minus']['crossover'].append([np.full((pop_size, 2), np.inf) for _ in range(num_pop)])
        gp.track['idx_minus']['crossover_hi'].append([np.full((pop_size, 2), np.inf) for _ in range(num_pop)])
        gp.track['idx_minus']['reproduction'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['idx_minus']['elite_isolated'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['idx_minus']['elite_ensemble'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['ensemble_idx'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['rank']['fitness']['isolated']['train'].append([np.full((pop_size), np.inf) for _ in range(num_pop)])
        gp.track['rank']['fitness']['isolated']['validation'].append([np.full((pop_size), np.inf) for _ in range(num_pop)])
        gp.track['rank']['fitness']['isolated']['test'].append([np.full((pop_size), np.inf) for _ in range(num_pop)])
        gp.track['rank']['fitness']['ensemble']['train'].append(np.full((pop_size), np.inf))
        gp.track['rank']['fitness']['ensemble']['validation'].append(np.full((pop_size), np.inf))
        gp.track['rank']['fitness']['ensemble']['test'].append(np.full((pop_size), np.inf))
        gp.track['rank']['complexity']['isolated'].append(np.full((pop_size, num_pop), np.inf))
        gp.track['rank']['complexity']['ensemble'].append(np.full((pop_size), np.inf))
        gp.track['all_ensemble']['idx'].append(None)
        gp.track['all_ensemble']['fitness']['train'].append(None)
        gp.track['all_ensemble']['fitness']['validation'].append(None)
        gp.track['all_ensemble']['fitness']['test'].append(None)
        
            
        
        
        
