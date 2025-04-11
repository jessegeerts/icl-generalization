from utils import dotdict as dd


config = dd(dict(
    model=dd(dict(
        h_dim=128,                      # hidden dimensionality of the transformer model
        n_heads=2,
        n_blocks=2,
        include_mlp=[False] * 2,
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=False,
        widening_factor=1,              # how much wider is the MLP hidden dim
        max_T=64,                       # max sequence length for the model
        out_dim=None,                   # note this is set later (dependent on N labels in data)
        drop_p=0.0,
        softmax_attn=[True, False],              # use softmax attention
        w_init_scale=0.002 / 2,
        include_query=True,             # include query in the input
    )),
    data=dd(dict(
        S=10000,
        K=2**10,                 # number of classes
        L=32,                   # number of labels
        D=63,                   # dimension of inputs
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
        learning_rate_adam=0.001, #1e-3,    # learning rate for Adam
        w_decay=1e-7,               # L2 regularisation parameter. note the torch implementation is a bit different from reddy jax code (it's multiplied by LR, so divide by LR to get desired w_decay param )
        niters=4000*30,
        optim='adam',                # 'adam' or 'sgd'
        grad_clip_value=.001
    )),
    log_to_wandb=False,
    logging_interval=1000,  # iterations
    save_weights=True,
    save_model=False,
    saving_interval=4000,
    model_dir='models/icl',
))
