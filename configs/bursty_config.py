from utils import dotdict as dd

config = dd(dict(
    model=dd(dict(
        h_dim=129,                      # hidden dimensionality of the transformer model
        n_heads=1,
        n_blocks=2,
        include_mlp=[False, False],
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=True,
        widening_factor=1,              # how much wider is the MLP hidden dim
        max_T=32,                       # max sequence length for the model
        out_dim=None,                    # note this is set later (dependent on N labels in data)
        drop_p=0.0
    )),
    data=dd(dict(
        type='symbolic',
        S=1600 * 10,  # number of examples (K * examples per class) (not used as we make our own data on the fly)
        n_rare_classes=1600-100-32,
        n_common_classes=32,
        n_holdout_classes=100,
        K=1024,                 # number of classes (needs to be divisible by L)
        L=32,                   # number of labels
        D=64,                   # dimension of inputs
        subD=32,                # dimension of subvectors (for partial exposure paradigm)
        alpha=0.,               # zipf exponent
        eps=0.75,                # within-class variance (higher => more ICL)
        Nmax=32,
    )),
    seq=dd(dict(
        ways=3,
        shots=4,
        N=3*4,                   # sequence length will be 2N + 1
        B=4,   # burstiness (currently moot)
        pB=1.,
        pC=1.,
        train_seq_type='bursty'
    )),
    train=dd(dict(
        batch_size=128,
        learning_rate=.001,
        w_decay=1e-5,           # L2 regularisation parameter. note the torch implementation is a bit different from reddy jax code (it's multiplied by LR, so divide by LR to get desired w_decay param )
        niters=100000
    )),
    log=dd(dict(
        log_to_wandb=True,
        logging_interval=100,  # iterations
        wandb_project="Burstiness",
    )),
    save_weights=True,
    save_model=False
))
