from utils import dotdict as dd

n_blocks = 3

config = dd(dict(
    model=dd(dict(
        h_dim=11,                      # hidden dimensionality of the transformer model
        n_heads=1,
        n_blocks=n_blocks,
        include_mlp=[False] * n_blocks,
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=False,
        widening_factor=1,              # how much wider is the MLP hidden dim
        max_T=64,                       # max sequence length for the model
        out_dim=None,                   # note this is set later (dependent on N labels in data)
        drop_p=0.0,
        softmax_attn=[False] * n_blocks,              # use softmax attention
        w_init_scale=0.002 / n_blocks,
        include_query=False,             # include query in the input
        weight_sharing=True,             # share weights across blocks
        dampening=.75,
    )),
    data=dd(dict(
        S=10000,
        K=2**10,                 # number of classes
        L=32,                   # number of labels
        D=10,                   # dimension of inputs
        alpha=0.,               # zipf exponent
        eps=0,                # within-class variance (higher => more ICL)
    )),
    seq=dd(dict(
        N=12,                   # sequence length will be 2N + 1
        Nmax=32,
        B=1,
        pB=1.,
        pC=1.,
        shuf=True,
        train_type='cat',
        no_repeats=False
    )),
    train=dd(dict(
        batch_size=2048,
        learning_rate=.01,          # learning rate for SGD
        learning_rate_adam=0.0005, #1e-3,    # learning rate for Adam
        w_decay=1e-7,               # L2 regularisation parameter. note the torch implementation is a bit different from reddy jax code (it's multiplied by LR, so divide by LR to get desired w_decay param )
        niters=10000 * 100,
        optim='adam',                # 'adam' or 'sgd'
        grad_clip_value=.001
    )),
    log_to_wandb=True,
    logging_interval=1000,  # iterations
    save_weights=True,
    save_model=False,
    saving_interval=4000,
    model_dir='models/icl',
))
