# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

<!-- insertion marker -->
## Unreleased

<small>[Compare with latest](https://github.com/saforem2/ezpz/compare/816746c41846b41b9a3c7fe0c00a00d5e4a4a90d...HEAD)</small>

### Added

- Add `docs/parallelism.md` ([90c40db](https://github.com/saforem2/ezpz/commit/90c40db0b9baeed8f5936196f3e2868bdb9e60f8) by Sam Foreman).
- Add ruff section to `pyproject.toml` ([06fcc91](https://github.com/saforem2/ezpz/commit/06fcc918b6a200156e1c23a327b982d8fced1ed3) by Sam Foreman).
- Add license to `pyproject.yaml` ([e2feee9](https://github.com/saforem2/ezpz/commit/e2feee92d21b07bf567730240b3b6b647a50d57e) by Sam Foreman).
- Add comms logger to `conf/ds_config.json` ([9b4d6bc](https://github.com/saforem2/ezpz/commit/9b4d6bc22bd949aee98302923859074472406e3d) by Sam Foreman).
- Add `h5py` to `pyproject.toml` ([abd4f48](https://github.com/saforem2/ezpz/commit/abd4f482e9769e57b25489e8a3b3d8cca8826732) by Sam Foreman).
- Add  `ezpz/dist.py:make_hostfile_from_slurm_env` ([09ca8e7](https://github.com/saforem2/ezpz/commit/09ca8e7b903930865f8ac445dd6adb246b5110bf) by Sam Foreman).
- Add `notes.md` ([fe969fe](https://github.com/saforem2/ezpz/commit/fe969febb795a0b512bd87550c3cb3d60afa7f32) by Sam Foreman).
- Add milliseconds to `src/ezpz/log/handler.py` ([999bdee](https://github.com/saforem2/ezpz/commit/999bdeefd5dba84d8ced3fb175c067e4e42e96e9) by Sam Foreman).
- Add `UTILS` to `configs.py` ([b30f511](https://github.com/saforem2/ezpz/commit/b30f511fbb9ad3eba5ee73da93c79a0ca930f60e) by Sam Foreman).
- Add `--cpu-bind depth -d 16` to `DIST_LAUNCH` in `bin/{save,get}jobenv` ([d9e334b](https://github.com/saforem2/ezpz/commit/d9e334b99d871a0afeb20f37852453e4b60e5ad7) by Sam Foreman).
- Add `plot` to `__init__.py` ([9b578bd](https://github.com/saforem2/ezpz/commit/9b578bd5c7e32f96103b1b6577d212a15e39b520) by Sam Foreman).
- Add `ezpz/profile.py` ([ba9a837](https://github.com/saforem2/ezpz/commit/ba9a83765c98000654123c44ab883cbf52380d49) by Sam Foreman).
- Add `src/ezpz/bin/utils.sh` ([8be0ec3](https://github.com/saforem2/ezpz/commit/8be0ec3269cf8a22f2cefbcd13cfd9dc768fa3bf) by Sam Foreman).
- Add timeout to deepspeed init ([1364e9e](https://github.com/saforem2/ezpz/commit/1364e9e66bf3870b42f549b7c8d704a68af18cdb) by Sam Foreman).
- Add `get_logger` method to `src/ezpz/configs.py` ([a9e4d30](https://github.com/saforem2/ezpz/commit/a9e4d307c66dea48bea3d685518cf08aff710c77) by Sam Foreman).
- Add `src/ezpz/utils.py` ([985ec4b](https://github.com/saforem2/ezpz/commit/985ec4b967af67b27a2487c0410e4b0835c900e8) by Sam Foreman).
- Add `src/ezpz/cria.py` ([1f65b9e](https://github.com/saforem2/ezpz/commit/1f65b9e1441e543c1b00c7d284aaba6ec626680d) by Sam Foreman).
- Add `src/ezpz/bin/affinity.sh` ([421d118](https://github.com/saforem2/ezpz/commit/421d118e249bb8db7c3ebf88c3743742d92f9c7b) by Sam Foreman).
- Add `test_dist.py` ([84737b2](https://github.com/saforem2/ezpz/commit/84737b26f506eb367fd8d95ad3a97da0dfc0fb36) by Sam Foreman).
- Add `src/ezpz/runtime.py` ([b576044](https://github.com/saforem2/ezpz/commit/b57604460f5cafe1c95377926157cbcd78521716) by Sam Foreman).
- Add `ezpz.dist.get_running_jobs_from_qstat()` ([9c29fbf](https://github.com/saforem2/ezpz/commit/9c29fbfc53895078d42f82fdedad4a933ad8a580) by Sam Foreman).
- Add `index.md` ([9d88cb2](https://github.com/saforem2/ezpz/commit/9d88cb2cd6cb8ef73daf4cde7766762aeb475152) by Sam Foreman).
- Add (empty) `setup.cfg` ([b17939c](https://github.com/saforem2/ezpz/commit/b17939cef8675fe3e1fb53c363d37381fcd492d9) by Sam Foreman).
- Add `{loadjobenv,savejobenv}.py` ([ace4c20](https://github.com/saforem2/ezpz/commit/ace4c206425e376a8a3bae0ac15de182e23e5149) by Sam Foreman).
- Add `train.py` ([9acc850](https://github.com/saforem2/ezpz/commit/9acc85034a1633945364c69e5d37b9c1bf48d955) by Sam Foreman).
- Add `startup_time` tracking to `__main__.py` ([5d8c580](https://github.com/saforem2/ezpz/commit/5d8c5804a1a3afeee95e80d699c4f1896c8caa4e) by Sam Foreman).
- Add `model.py` ([d3748f5](https://github.com/saforem2/ezpz/commit/d3748f5cb630752ef36407f03b7edd72b2497597) by Sam Foreman).
- Add `jobs.py` ([106303d](https://github.com/saforem2/ezpz/commit/106303d00626f8aa3a5b185df036e28d244539bb) by Sam Foreman).
- Add timing info to `__main__.py` ([c6c9f99](https://github.com/saforem2/ezpz/commit/c6c9f991d4091e1ae53074dee8f49be1af1cc653) by Sam Foreman).
- Add `test.py` ([0c2a5ba](https://github.com/saforem2/ezpz/commit/0c2a5ba0bfd41f0c2df076eebcb77319bb1519f0) by Sam Foreman).
- Add `history.py` ([8777717](https://github.com/saforem2/ezpz/commit/8777717f28dc0e47dc05b4817fe367f8ab49b539) by Sam Foreman).
- Add `plot.py` ([35a571e](https://github.com/saforem2/ezpz/commit/35a571e3164959ecb7d7de5f9e4b748b28375bfb) by Sam Foreman).
- Add `conf/hydra/job_logging/custom.yaml` ([51aa6de](https://github.com/saforem2/ezpz/commit/51aa6de13254718a3631ab1ad220b72d4c1a2402) by Sam Foreman).
- Adds `conf/hydra/job_logging/rich.yaml` ([af62bfd](https://github.com/saforem2/ezpz/commit/af62bfd79dcef8d64f47797d197f95d450e900c7) by Sam Foreman).
- Adds `start_method` to `setup_wandb(...)` in `dist.py` ([5117f96](https://github.com/saforem2/ezpz/commit/5117f96491aa78cb51c7fc0b224ef5ab55663e92) by Sam Foreman).
- Adds `conf/ds_config.yaml` ([3cdac42](https://github.com/saforem2/ezpz/commit/3cdac42a8ff8660dfc05f9e34577004576e5d852) by Sam Foreman).
- Adds `{train.py,configs.py,conf/*}` w/ Hydra support ([54a78b6](https://github.com/saforem2/ezpz/commit/54a78b680d8ec74ede55e94b52518105a7ae1aa3) by Sam Foreman).
- Add `bin/{setup.sh,train.sh}` ([01e91e2](https://github.com/saforem2/ezpz/commit/01e91e2b8ebd4f246fc40bbd1dcaadd3dc3f5d9d) by Sam Foreman).
- Add `src/ezpz/__main__.py`, update `__init__.py` ([fbf6151](https://github.com/saforem2/ezpz/commit/fbf615175088c27f9738f8bd58294dc0efcb891e) by Sam Foreman).
- Adds `ezpz/bin/*.sh` ([8686bef](https://github.com/saforem2/ezpz/commit/8686bef6de66544a494e6c51ff2d7f5737575bfa) by Sam Foreman).
- Adds `setup_tensorflow` to `ezpz/dist.py` ([03edbbe](https://github.com/saforem2/ezpz/commit/03edbbe0ecfd255fa75558caf1b5afed7772f0d6) by Sam Foreman).
- Add `src/ezpz/*` ([0dcb423](https://github.com/saforem2/ezpz/commit/0dcb42360ae6c1a14efd29d8b8f5197124348a1c) by Sam Foreman).
- Adds `pyproject.toml` ([c5a5700](https://github.com/saforem2/ezpz/commit/c5a57001cc4b82cb4ac5ec9b9c2b2a803c766ba6) by Sam Foreman).

### Fixed

- fix: Fix bug in `ezpz/launch.py` ([e2b2b57](https://github.com/saforem2/ezpz/commit/e2b2b5760e28a1d39abe5d68a2f51ecca53ebbcb) by Sam Foreman).
- fix: Fix imports on Polaris ([f33d8d2](https://github.com/saforem2/ezpz/commit/f33d8d2e9b7e19699f3c4781f8a6f0427ded4bab) by Sam Foreman).
- fix: Fix bug in `ezpz/dist.py` ([7bdf4ac](https://github.com/saforem2/ezpz/commit/7bdf4ac2bb8e5e0cdde111939cfa6f2a3335f1c9) by Sam Foreman).
- fix: Fix `pyproject.toml` ([473b5fb](https://github.com/saforem2/ezpz/commit/473b5fbdc45c8c7e1eba288dc066eec510032f8d) by Sam Foreman).
- fix: Fix backend selection on XPU device ([b86dacf](https://github.com/saforem2/ezpz/commit/b86dacfc7cc895bfca16b8218145d815d7833ca6) by Sam Foreman).
- fix: Fix torch backend on XPU ([487cab9](https://github.com/saforem2/ezpz/commit/487cab9c0386a160a0041dbba883dd1ceae7c258) by Sam Foreman).
- fix: Fix imports in `tp/__init__.py` on Aurora ([027d152](https://github.com/saforem2/ezpz/commit/027d15245d772932c58f256d6b95674bdae00f08) by Sam Foreman).
- fix: Fix broken import on Polaris ([6e15551](https://github.com/saforem2/ezpz/commit/6e155518eb99557f276b407afd968f0c5f19066c) by Sam Foreman).
- fix: Update `src/ezpz/bin/utils.sh` w/ new conda on Sunspot ([baa4471](https://github.com/saforem2/ezpz/commit/baa44710aac997fede6de0e4b2f41d495b277e99) by Sam Foreman).
- fix jobenv does not exist when passing single argument ([c8f1194](https://github.com/saforem2/ezpz/commit/c8f1194bb4fbee318bb0e644cb708a5e35251a66) by Ray Andrew).
- Fix `mpi4py` init bug in base `2024-04-29` on Polaris ([c83ad21](https://github.com/saforem2/ezpz/commit/c83ad218f45c25ba9d45e0669dc1e88a518b39c7) by Sam Foreman).
- Fix `ccl` issue in in `dist.py` ([e7e1862](https://github.com/saforem2/ezpz/commit/e7e18621f4f5ed7e22a27cc2a880d3bf6b986030) by Sam Foreman).
- Fix python interface when no scheduler present ([342a29c](https://github.com/saforem2/ezpz/commit/342a29cef718db21efb3e6876c8854f296a6695a) by Sam Foreman).
- Fix import in `__init__.py` ([65bba02](https://github.com/saforem2/ezpz/commit/65bba02af3cf34261794c7c6852ebc7b524d26ec) by Sam Foreman).

### Removed

- Remove `torch.use_deterministic_algorithms(True) from `dist.py` ([5216f35](https://github.com/saforem2/ezpz/commit/5216f35261e44364ee42658b6c5e3aa93481a418) by Sam Foreman).
- Remove `build.targets.{sdist,wheel}` from `pyproject.toml` ([1d1eb2b](https://github.com/saforem2/ezpz/commit/1d1eb2b242816a8f70be21d3066ef540bb4ba714) by Sam Foreman).
- Remove `matplotx` from `history.py` ([7991580](https://github.com/saforem2/ezpz/commit/799158052d6badbe4ba701b085170994f746020a) by Sam Foreman).
- Remove unnecessary `test.py` ([23ea0cb](https://github.com/saforem2/ezpz/commit/23ea0cbc60ef324ae792264590daa734407fe388) by Sam Foreman).

<!-- insertion marker -->
