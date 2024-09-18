from utils import dotdict as dd

config = dd(dict(
    model=dd(dict(
        pos_dim=512,                      # hidden dimensionality of the transformer model
        emb_dim=512,                    # dimensionality of the input embeddings
        n_heads=4,
        n_blocks=2,
        include_mlp=False,               # include MLP after attention block (this can also be a list of len n_blocks)
        activation='relu',              # activation fn for the MLP
        n_mlp_layers=None,              # TODO: make this mutable
        apply_ln=True,
        widening_factor=4,              # how much wider is the MLP hidden dim
        max_T=32,                       # max sequence length for the model
        out_dim=None,                   # note this is set later (dependent on N labels in data)
        drop_p=0.1,                     # dropout probability
        pos_emb_type='sinusoidal',          # type of position embedding
        pos_emb_loc='append',               # add or append to token embeddings
        prediction_mode = 'regress'
    )),
    data=dd(dict(
        K=1623
    )),
    seq=dd(dict(
        N=7,                        # number of examples in the sequence
        repeats=1,
        train_type='IW',            # type of training sequence (IW or IC)
    )),
    train=dd(dict(
        batch_size=1,
        learning_rate=.0001,
        w_decay=1e-5,           # L2 regularisation parameter. note the torch implementation is a bit different from reddy jax code (it's multiplied by LR, so divide by LR to get desired w_decay param )
        niters=5000
    )),
    log=dd(dict(
        log_to_wandb=True,
        logging_interval=40,  # iterations
        wandb_project="TransitiveInference",
    )),
    save_weights=True,  # save attention weights for logging purposes
    save_model=False    # save model weights for later reuse
))
