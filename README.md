# Bliss Learner

Rust binary that learns a personalised Mahalanobis distance matrix from
user-provided "odd-one-out" training triplets stored in a bliss SQLite database.
The learned matrix is used by [Bliss Mixer](https://github.com/chrober/bliss-mixer)
to improve song similarity when dynamic weighting is enabled.

This is a Rust port of [bliss-metric-learning](https://github.com/Polochon-street/bliss-metric-learning)
by Polochon-street. The core algorithm (crowd-kernel / STE triplet loss with
probit model, L-BFGS-B optimisation, 5-fold cross-validation over a lambda grid)
is faithfully preserved; the implementation replaces Python/numpy with Rust/ndarray
so it can run as a standalone binary without any runtime dependencies.

Intended to be used together with the [Bliss LMS DSTM plugin](https://github.com/CDrummond/lms-blissmixer),
which provides a similarity survey UI to collect training triplets and triggers
the learning process from its settings page.


## Building

[Rust](https://www.rust-lang.org/tools/install) is required to build.

Build with `cargo build --release`


## Usage

```
bliss-learner --db /path/to/bliss.db --triplets /path/to/training_triplets.json --output /path/to/learned_matrix.json
```

### Options

| Flag | Description |
|------|-------------|
| `-d`, `--db` | Path to bliss.db (analysis features) |
| `-t`, `--triplets` | Path to training_triplets.json (survey data) |
| `-o`, `--output` | Output path for learned_matrix.json |
| `-L`, `--lms` | LMS hostname (default: 127.0.0.1) |
| `-J`, `--json` | LMS JSON-RPC port (default: 9000) |
| `-N`, `--notifs` | Send progress notifications to LMS |
| `-l`, `--logging` | Log level (default: info) |

When `--notifs` is enabled, progress messages are pushed to LMS via JSON-RPC so
the plugin settings page can display live status updates.


## Credits

The metric learning algorithm is based on
[bliss-metric-learning](https://github.com/Polochon-street/bliss-metric-learning)
by [Polochon-street](https://github.com/Polochon-street), which itself builds on
the crowd-kernel (STE) model described in:

> Tamuz, O., Liu, C., Belongie, S., Shamir, O., & Kalai, A. T. (2011).
> *Adaptively Learning the Crowd Kernel.* ICML.
