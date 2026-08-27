# hirax-solar-zt

Combined-day, all-frequency 1D Zernike ("ZT") polynomial fit of the HIRAX
solar-transit amplitude. `dataset.h5` holds the fit results for all 81
HIRAX frequency channels (0.8 -> 0.4 GHz). It's not included in this
repo -- the author shares it separately with collaboration members -- but
once you have it, you don't need to build anything: just drop it in this
directory and go straight to plotting.

## Setup

Create a venv named `hiraxsolar` so it's obvious which environment this
repo's dependencies live in, then install into it:

```bash
python3 -m venv hiraxsolar
source hiraxsolar/bin/activate   # Windows: hiraxsolar\Scripts\activate

pip install -r requirements.txt
```

## Plot a channel

```bash
python quickstart.py dataset.h5 --freq-idx 40
```

This opens a two-panel plot for channel 40 (~600 MHz):
- **top**: the across-day mean amplitude (black points, with day-to-day
  scatter as error bars) vs. the fitted Zernike ("ZT") curve (red)
- **bottom**: the fit residuals

Drop `--freq-idx` to get the middle channel by default:

```bash
python quickstart.py dataset.h5
```

Valid `--freq-idx` values are `0`-`80` (channel 0 = 800 MHz, channel 80 =
400 MHz -- frequency decreases as the index increases). `quickstart.py`
prints the fit diagnostics (`x0`, `sigma`, `N_full`, `chi2_red`, `n_valid`)
for whichever channel it plots.

## Not happy with a channel's fit?

Each channel's polynomial order (`N_full`) and width (`sigma`) already
stored in `dataset.h5` came from an automated two-stage search -- optimize
`sigma` at a fixed probe `N_full`, then scan `N_full` at that fixed
`sigma` -- which isn't a true joint minimum. If a channel's fit looks
overfit, underfit, or otherwise off in the plot, use `refit_channel.py` to
play with `N_full` and `sigma` yourself:

```bash
# see chi2_red for a range of N_full choices at the channel's stored sigma
python refit_channel.py dataset.h5 --freq-idx 40 --list

# ... or at a sigma you choose
python refit_channel.py dataset.h5 --freq-idx 40 --list --sigma 0.5

# try an (N_full, sigma) pair, look at the plot -- dataset.h5 is untouched so far
python refit_channel.py dataset.h5 --freq-idx 40 --n 300 --sigma 0.5

# happy with it? write it back
python refit_channel.py dataset.h5 --freq-idx 40 --n 300 --sigma 0.5 --write

# ... or write into a different file instead, leaving dataset.h5 untouched
python refit_channel.py dataset.h5 --freq-idx 40 --n 300 --sigma 0.5 --write --out dataset_refit.h5
```

This only refits against the `mean`/`var`/`valid`/`x0` already stored for
that channel -- it doesn't touch the raw visibility data or any other
channel's group. `--write` overwrites `coef`, `model`, `N_full`, `sigma`,
`sigma_0`, and `chi2_red` for just that one `freq_{idx:03d}` group.

By default `--write` overwrites `h5_path` in place. Pass `--out` with a
different filename to write there instead: if that file doesn't already
exist, `h5_path` is copied to it first (so `h5_path` itself is never
touched), then just the one channel is updated on the copy. Passing `--out`
with the same filename as `h5_path` is the same as omitting it. Rerunning
with the same `--out` file keeps refitting more channels onto that same
copy without re-copying each time.

### [beta] Joint (N_full, sigma) search

Instead of picking `N_full`/`sigma` by hand, `--joint` searches for the
minimum chi2_red over both at once: for each candidate `N_full` it
optimizes `sigma` continuously, then takes the best `(N_full, sigma)` pair
across all candidates. This is more thorough than the two-stage search that
originally produced `dataset.h5`, but it's slower (a continuous
optimization per candidate `N_full`, ~11 of them) and still scans `N_full`
discretely since it isn't a continuous parameter -- treat it as a starting
point to sanity check by eye, not a final answer:

```bash
python refit_channel.py dataset.h5 --freq-idx 40 --joint
python refit_channel.py dataset.h5 --freq-idx 40 --joint --write
```

## HDF5 schema

Top-level:

| Path | Description |
|---|---|
| `dt_grid` | `(n_grid,)` float, hours from transit -- shared time axis for every channel |
| `freqs_ghz` | `(81,)` float, the HIRAX channel frequencies in GHz |

Per channel, group `freq_{idx:03d}` (e.g. `freq_040`):

| Path | Shape | Description |
|---|---|---|
| `mean` | `(n_grid,)` | across-day mean of `log10(smoothed \|vis\|)` at each `dt_grid` point (NaN where no day covers that point) |
| `var` | `(n_grid,)` | across-day variance at each `dt_grid` point |
| `valid` | `(n_grid,)` bool | `True` where `mean`/`var` are finite and `var > 0` -- the points actually used in the fit |
| `model` | `(n_grid,)` | fitted Zernike model evaluated on the *full* `dt_grid` (using the fitted coefficients and basis) |
| `coef` | `(n_modes,)` | fitted Zernike coefficients (Noll-ordered, length depends on how many modes were valid at this `N_full`) |
| attrs: `freq_ghz`, `x0`, `sigma_gauss`, `sigma_0`, `sigma`, `N_full`, `chi2_red`, `n_valid` | fit parameters and diagnostics for this channel |

`mean`/`var`/`valid`/`model` are all stored at the same `dt_grid` length, so
you can index and plot them together without re-deriving the fit mask.

**Caution:** `model` is the fitted Zernike polynomial evaluated across the
*entire* `dt_grid`, including points outside `valid` (where no day actually
had data). High-`N_full` polynomial fits can extrapolate wildly just past
the edge of the fitted domain -- see the edges of `quickstart.py`'s plot.
Mask with `valid` before using `model` for anything quantitative; it's left
unmasked in storage only so plotting doesn't require re-deriving the fit
domain.

## License

Not yet decided -- treat as all-rights-reserved until a LICENSE file is
added.
