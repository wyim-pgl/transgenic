# MVP runtime read-only preflight — 2026-09-15 PDT

DeltaAI authentication restored by author. At 15:26 PDT seed123 next job3131537 PENDING Priority, later links Dependency. No checkout or job mutations. repo_seeds HEAD8f049eb3ba72ba7541076fe9fc8f6019b90c0f7f.

Read-only filesystem inventory resolved best -> /work/nvme/bilv/wcyim/transgenic/runs/seed123/epoch_04. This concrete directory, not mutable best, is a candidate initializer; author approval of loss semantics does not certify checkpoint contents or initiate GPU use.

- epoch_04/model.safetensors 2091010540B; separate accelerate_state/model.safetensors2091005908B. Do not confuse weights-only export with resume state.
- epoch04 meta: epoch4, global_step8284, step0, pending_micro12, seed123, best_eval_score0.6525425314903259.
- eval: eval_loss0.6525425314903259, train_loss0.6324494481086731, is_best true, bad_epochs0, patience3.
- Parent run_config.json2384B contains recipe (no config.json observed in checkpoint): vocab290, encoder768/decoder1536, encoder/decoder12layers, heads6, decoder_ffn3072, decoderpositions8192, encoderlimit129024, dropout0.1, LongSafari/hyenadna-large-1m-seqlen-hf. Fresh model must audit full semantic recipe rather than trusting checkpoint filename.
- Parent runtime_identity.txt recorded SIF transgenic-ngc-26.08.sif, SHA2561a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6, repo8f049eb, gpus4/cpus64, torch2.14.0a0+4fdf77b940.nv26.08, nonreentrant checkpointing, compile on. This is a recorded hash, NOT freshly rehashed image identity. A shallow find did not locate SIF (actual path still to confirm).

A first metadata-read command had shell quoting SyntaxError before file access; replaced with stdin Python heredoc. No model loading/unpickling or remote writes occurred.

At ~15:32 PDT CPU queue advanced: Osativa no longer in squeue (completion accounting/source gate not yet checked), Ptrichocarpa running, Ppatens running, Sbicolor/Vvinifera waiting. full DB6164118 running1h06m; no final success files, partial DB/WAL untouched. These are snapshots, not a monitor.
