from pprint import pprint

from util.clients import get_me_clients

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants
bot_name = bot_specific_constants.blogName


# TODO: DRY (centralize client managers)
class ClientPool:
    def __init__(self, min_remaining_hour=10, min_remaining_day=10):
        self.min_remaining_hour = min_remaining_hour
        self.min_remaining_day = min_remaining_day

        self.private_clients = get_me_clients(include_dashboard=False)
        self.dashboard_clients = get_me_clients(include_private=False)

    def is_group_available(self, clients):
        ratelimit_data = []
        for client in clients:
            try:
                ratelimit_data.append(client.get_ratelimit_data())
            except KeyError:
                # already used up
                pass

        day_remaining = sum([rd["day"]["remaining"] for rd in ratelimit_data])
        hour_remaining = sum([rd["hour"]["remaining"] for rd in ratelimit_data])

        ok_day = day_remaining > self.min_remaining_day
        ok_hour = hour_remaining > self.min_remaining_hour

        return ok_day and ok_hour

    def client_name(self, client):
        for i, cl in enumerate(self.private_clients):
            if client is cl:
                return f"p{i}"

        for i, cl in enumerate(self.dashboard_clients):
            if client is cl:
                return f"d{i}"

        return "unk"

    def is_client_available(self, client):
        return self.is_group_available([client])

    def _group_agg(self, clients, key, key2=None):
        values = []
        for client in clients:
            if not self.is_client_available(client):
                value = 0
            else:
                try:
                    value = client.get_ratelimit_data()[key]
                    if key2:
                        value = value[key2]
                except KeyError:
                    value = 0
            values.append(value)

        return values

    def _group_rates(self, clients):
        return self._group_agg(clients, 'effective_max_rate')

    def _group_remaining(self, clients):
        return self._group_agg(clients, 'day', key2="remaining")

    def pick_from_group(self, clients):
        rates = self._group_rates(clients)

        if len(rates) == 0:
            raise ValueError(f"ratelimit exhausted in group")

        ixs = list(range(len(clients)))
        ix_max = sorted(ixs, key=lambda ix: rates[ix])[-1]

        choice = clients[ix_max]

        return choice

    def _get_group(self, client_type='any'):
        if client_type == 'any':
            group = self.clients
        elif client_type == 'private':
            group = self.private_clients
        elif client_type == 'dashboard':
            group = self.dashboard_clients
        else:
            raise ValueError(client_type)
        return group

    def remaining(self, client_type='any'):
        group = self._get_group(client_type)
        return sum(self._group_remaining(group))

    def max_rate(self, client_type='any'):
        group = self._get_group(client_type)
        return sum(self._group_rates(group))

    def pick_group(self):
        if not self.is_group_available(self.private_clients):
            return self.dashboard_clients
        if not self.is_group_available(self.dashboard_clients):
            return self.private_clients

        private_rates = self._group_rates(self.private_clients)
        dash_rates = self._group_rates(self.dashboard_clients)

        if sum(private_rates) > sum(dash_rates):
            return self.private_clients
        return self.dashboard_clients

    def get_client(self):
        return self.pick_from_group(self.pick_group())

    def get_private_client(self):
        return self.pick_from_group(self.private_clients)

    def get_dashboard_client(self):
        return self.pick_from_group(self.dashboard_clients)

    def report(self):
        for i, client in enumerate(self.private_clients):
            print(f"Private {i}:")
            pprint(client.get_ratelimit_data())

        for i, client in enumerate(self.dashboard_clients):
            print(f"Dash {i}:")
            pprint(client.get_ratelimit_data())

    @property
    def clients(self):
        return self.private_clients + self.dashboard_clients

    def compute_checkprob(self, requests_per_check: int, time_per_cycle: float, verbose=False, client_type='any'):
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        summed_max_rate = self.max_rate(client_type)

        vprint(f"summed_max_rate: {summed_max_rate:.4f}")

        # if we checked *every* cycle, we could support up this many requests per check
        requests_per_cycle = time_per_cycle * summed_max_rate
        vprint(f"requests_per_cycle: {requests_per_cycle:.4f} (vs requests_per_check {requests_per_check:.4})")

        # we'll check only a fraction `checkprob` of the cycles, to support up to `requests_needed_to_check`
        checkprob = requests_per_cycle / max(requests_per_check, 1)

        return checkprob
