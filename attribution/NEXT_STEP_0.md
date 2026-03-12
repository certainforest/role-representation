I now want to replicate the llama experiemtn on Qwen/Qwen3-8B.
I moved all prev plots to attribution_plot_save because its geting messy. we will rerun experiments cleanly.

I need to code that clearly produces the plots i want, following the IOI notebook. Look at @/mnt/ssd/aryawu/role-representation/attribution/LLAMA_RESULT_IOISTYLE_INTERPRETATION.md to understand what each plot means.

I want a new piece of code that systematically does the following. 
I want a utils.py that loads the experiments. I think i would want to be able to load all 8 so that we can clearly separate, when voting, which heads are binding-id-resolving specific, which heads are person name specific, which heads are attribute(country/sports) specific. And then i think i would need to save all these heads, for subsequent circuit analysis, which we should put aside for now.

Phase 1: 
1_logit_lens.png
2_per_layer_attribution.png
3_per_head_attribution.png
4_attn_analysis.png: We visualize the top 3 positive and negative heads by direct logit attribution. 

Phase 2:
5_patch_resid_pre.png
6_patch_all_blocks.png
7_patch_heads_all_pos.png
8_patch_qkv.png


