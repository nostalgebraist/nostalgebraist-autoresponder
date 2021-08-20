from datetime import datetime, timedelta, time as dtime
from util.times import TZ_PST

BASE_SLOWDOWN_LEVEL = {"name": "base", "rate_ratio_thresh": 1., "n_remaining_thresh": 60, "SLEEP_TIME_scale": 1.,
                       "MAX_POSTS_PER_STEP_scale": 1.}

SLOWDOWN_LEVELS = [
    BASE_SLOWDOWN_LEVEL,
    {"name": "slower", "rate_ratio_thresh": 1.5, "n_remaining_thresh": 40, "SLEEP_TIME_scale": 2.5,
     "MAX_POSTS_PER_STEP_scale": 3.1 / 5},
    {"name": "slower2", "rate_ratio_thresh": 2.5, "n_remaining_thresh": 25, "SLEEP_TIME_scale": 5,
     "MAX_POSTS_PER_STEP_scale": 2.1 / 5},
    {"name": "slowest", "rate_ratio_thresh": 1000, "n_remaining_thresh": 0, "SLEEP_TIME_scale": 10,
     "MAX_POSTS_PER_STEP_scale": 1.1 / 5},
]

HARDSTOP_SLOWDOWN_LEVEL = {
    "name": "hardstop", "rate_ratio_thresh": 1000, "n_remaining_thresh": 0, "SLEEP_TIME_scale": 1,
    "MAX_POSTS_PER_STEP_scale": 0
}


HARDSTOP_AT_N_REMAINING = 1


def post_limit_reset_ts(now=None):
    # this assumes:
    #   - tumblr resets at midnight EST
    #   - frank is running in PST
    # TODO: revisit this if i'm on vacation or something

    if now is None:
        now = datetime.now(tz=TZ_PST).replace(tzinfo=None)

    one_day_ago = now - timedelta(days=1)

    reset_date = now.date() if now.hour >= 21 else one_day_ago.date()

    reset_ts = datetime.combine(reset_date, dtime(hour=21))

    return reset_ts


def count_posts_since_ts(post_payloads, ts):
    is_after = [
        (datetime.fromtimestamp(entry['timestamp']) - ts).total_seconds() > 0
        for entry in post_payloads
        if not entry.get('is_pinned')
    ]

    if all(is_after):
        msg = f"count_posts_since_ts: all {len(is_after)} posts passed are after ts passed."
        msg += " Count returned will be a lower bound."
        print(msg)

    return sum(is_after)


def count_posts_since_reset(post_payloads, now=None):
    reset_ts = post_limit_reset_ts(now=now)

    return count_posts_since_ts(post_payloads, ts=reset_ts)


def compute_max_rate_until_next_reset(post_payloads, now=None, max_per_24h=250):
    if now is None:
        now = datetime.now(tz=TZ_PST).replace(tzinfo=None)

    reset_ts = post_limit_reset_ts(now=now)

    posts_since_last_reset = count_posts_since_ts(post_payloads, ts=reset_ts)

    n_remaining = max_per_24h - posts_since_last_reset

    next_reset_ts = reset_ts + timedelta(days=1)
    time_until_next_reset = next_reset_ts - now
    seconds_until_next_reset = time_until_next_reset.total_seconds()

    max_rate = n_remaining / seconds_until_next_reset
    return max_rate


def compute_rate_over_last_hours(post_payloads, avg_over_hours, now=None):
    if now is None:
        now = datetime.now(tz=TZ_PST).replace(tzinfo=None)

    delt = timedelta(hours=avg_over_hours)
    ts = now - delt
    n = count_posts_since_ts(post_payloads, ts)
    rate = n / delt.total_seconds()

    return n, rate


def review_rates(post_payloads, max_per_24h=250, hour_windows=(1, 2, 4, 12,), now=None, max_rate=None):
    if not max_rate:
        max_rate = compute_max_rate_until_next_reset(post_payloads, now=now, max_per_24h=max_per_24h)

    if now is None:
        now = datetime.now(tz=TZ_PST).replace(tzinfo=None)

    reset_ts = post_limit_reset_ts(now=now)

    is_since_reset = [False for _ in hour_windows]
    hour_windows += (((now - reset_ts).total_seconds() / 3600),)
    is_since_reset.append(True)

    ns = []
    rates = []
    for h in hour_windows:
        n, rate = compute_rate_over_last_hours(post_payloads, avg_over_hours=h, now=now)

        ns.append(n)
        rates.append(rate)

    for h, n, r, isr, in zip(hour_windows, ns, rates, is_since_reset):
        ratio = r / max_rate
        pieces = [f"last {float(h):<4.1f} hours:", f"{n:<3} posts", f"{ratio:<6.1%} of max rate"]

        if isr:
            pieces = ["[since reset]"] + pieces
        else:
            pieces = [""] + pieces

        msg = "".join([f"{piece:<20}\t" for piece in pieces])
        print(msg)


def select_slowdown_level(post_payloads, avg_over_hours=2, max_per_24h=250, hardstop_pad=0, ref_level=None, now=None,
                          verbose=True):
    max_rate = compute_max_rate_until_next_reset(post_payloads, now=now, max_per_24h=max_per_24h)

    _, rate = compute_rate_over_last_hours(post_payloads, avg_over_hours=avg_over_hours, now=now)
    n_since_reset = count_posts_since_reset(post_payloads, now=now)

    ratio = rate / max_rate
    n_remaining = max_per_24h - n_since_reset

    selected = None
    for level in SLOWDOWN_LEVELS:
        if (ratio <= level["rate_ratio_thresh"]) and (n_remaining > level["n_remaining_thresh"]):
            selected = level
            break
    if selected is None:
        selected = sorted(SLOWDOWN_LEVELS, key=lambda d: d["rate_ratio_thresh"])[-1]

    hardstopping = n_remaining <= (HARDSTOP_AT_N_REMAINING + hardstop_pad)

    if hardstopping:
        selected = HARDSTOP_SLOWDOWN_LEVEL

    if verbose:
        print()
        review_rates(post_payloads, now=now, max_rate=max_rate)
        print(
            f"\nselected level {repr(selected['name'])} based on {avg_over_hours}-hour ratio {ratio:.1%}, n_remaining={n_remaining}")

        if ref_level and ref_level['name'] != selected['name']:
            print(f"SWITCHED from {ref_level['name']} to {selected['name']}\n")

    return selected
