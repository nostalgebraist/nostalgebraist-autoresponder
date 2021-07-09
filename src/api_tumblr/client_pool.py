from pprint import pprint

from util.clients import get_me_clients

from config.bot_config import BotSpecificConstants
bot_name = BotSpecificConstants.load().blogName


# TODO: DRY (centralize client managers)
class ClientPool:
    def __init__(self, min_remaining_hour=100, min_remaining_day=300):
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

    def _group_rates(self, clients):
        clients = [client for client in clients if self.is_client_available(client)]

        if len(clients) == 0:
            return

        rates = []
        for client in clients:
            try:
                rates.append(client.get_ratelimit_data()['effective_max_rate'])
            except KeyError:
                rates.append(0)

        return rates

    def pick_from_group(self, clients):
        rates = self._group_rates(clients)

        if len(rates) == 0:
            raise ValueError(f"ratelimit exhausted in group")

        ixs = list(range(len(clients)))
        ix_max = sorted(ixs, key=lambda ix: rates[ix])[-1]

        return clients[ix_max]

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