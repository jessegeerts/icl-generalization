from utils import dotdict as dd


config = dd(dict(
    model=dd(dict(
        pos_dim=64,
        emb_dim=64,
        n_heads=4,
        n_blocks=2,
        include_mlp=False,
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=True,
        widening_factor=1,              # how much wider is the MLP hidden dim
        max_T=32,                       # max sequence length for the model
        out_dim=None,                    # note this is set later (dependent on N labels in data)
        drop_p=0.2,
        pos_emb_loc='none',  # in this experiment we add position embeddings already in the embedder class
        prediction_mode='regress',
        pos_emb_type='sinusoidal',  # in this experiment we add position embeddings already in the embedder class (sinusoidal or onehot)
    )),
    data=dd(dict(
        S=1603,
        n_rare_classes=1600-100-32,
        n_common_classes=32,
        n_holdout_classes=100,
        K=1602,                 # number of classes (needs to be divisible by L)
        L=2,                   # number of labels
        D=64,                   # dimension of inputs
        subD=32,                # dimension of subvectors (for partial exposure paradigm)
        alpha=0.,               # zipf exponent
        eps=0.75,                # within-class variance (higher => more ICL)
        Nmax=32,
    )),
    seq=dd(dict(
        ways=2,                  # number of classes in a few-shot task
        shots=1,
        N=None,  # (ways*shots) sequence length will be 3N + 2 (note this must be at least ways*shots, actually currently exactly ways*shots)
        B=4,
        pB=1.,
        pC=1.,
        train_seq_type='fewshot'
    )),
    train=dd(dict(
        batch_size=128,
        learning_rate=0.00036,
        w_decay=7.091481879812184e-05,           # L2 regularisation parameter
        lr_scheduler='warmup_constant',
        warmup_steps=3000,
        niters=30000
    )),
    log=dd(dict(
        log_to_wandb=True,
        logging_interval=100,  # iterations
        wandb_project="in-context-trans-inf-hyperparam-search",
    )),
    save_weights=True,
    save_model=False
))
