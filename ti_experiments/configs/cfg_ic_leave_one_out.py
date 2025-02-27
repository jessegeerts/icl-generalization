from utils import dotdict as dd


config = dd(dict(
    model=dd(dict(
        pos_dim=0,  # positional encoding dimensionality (we use rope)
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
        pos_emb_type='onehot',  # in this experiment we add position embeddings already in the embedder class (sinusoidal or onehot)
        pos_emb_randomization='no_shift',
        add_pos_encodings=True
    )),
    data=dd(dict(
        D=64,  # input dimensionality
    )),
    seq=dd(dict(
        ways=7,                  # number of classes in a few-shot task
        shots=1,
        N=None,  # (ways*shots) sequence length will be 3N + 2 (note this must be at least ways*shots, actually currently exactly ways*shots)
        B=4,
        train_seq_type='ic',
        leave_one_out=True,  # false means copy of the query is in context during training
        include_distal_in_training=False,  # we train only on adjacent pairs
        add_distractors_for_iw_seqs=False
    )),
    train=dd(dict(
        batch_size=128,
        learning_rate=0.00000987608117559501,
        w_decay=0.00004325692307609201,           # L2 regularisation parameter
        lr_scheduler='warmup_constant',
        warmup_steps=3241,
        niters=90000,
        steps_above_criterion=10,
    )),
    log=dd(dict(
        log_to_wandb=True,
        logging_interval=200,  # iterations
        checkpoint_interval=4000,  # iterations
        checkpoint_dir='checkpoints',
        wandb_project="in-context-trans-inf-hyperparam-search",
        run_name=None
    )),
    save_weights=True,
    save_hiddens=False,
    save_model=False,
    eval_at_all_distances=True
))
