'''
    collect demonstrations for ten different tasks, including:
        'reach-v2',
        'button-press-v2',
        'drawer-open-v2',
        'door-open-v2',
        'sweep-v2', 
        'push-v2', 
        'sweep-into-v2',
        'coffee-button-v2',
        'faucet-open-v2',
        'window-open-v2'
    for each task, we collect 100 rollouts
'''

from utils import collect_reachv2, collect_buttonpressv2, collect_draweropenv2, collect_dooropenv2, collect_sweepv2, collect_pushv2, collect_sweepintov2, collect_coffeebuttonv2, collect_faucetopenv2, collect_windowopenv2

collect_reachv2(collect_again=True, rgb=True)
collect_buttonpressv2(collect_again=True, rgb=True)
collect_draweropenv2(collect_again=True, rgb=True)
collect_dooropenv2(collect_again=True, rgb=True)
collect_sweepv2(collect_again=True, rgb=True)
collect_pushv2(collect_again=True, rgb=True)
collect_sweepintov2(collect_again=True, rgb=True)
collect_faucetopenv2(collect_again=True, rgb=True)
collect_windowopenv2(collect_again=True, rgb=True)
collect_coffeebuttonv2(collect_again=True, rgb=True)
