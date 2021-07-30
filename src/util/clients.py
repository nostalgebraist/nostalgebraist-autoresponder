from config.bot_config import BotSpecificConstants


def get_me_clients(npf=True, include_private=True, include_dashboard=True):
    import config.bot_config_singleton
    bot_specific_constants = config.bot_config_singleton.bot_specific_constants

    clients = []
    if include_private:
        clients += bot_specific_constants.private_clients
    if include_dashboard:
        clients += bot_specific_constants.dashboard_clients

    if npf:
        from tumblr_to_text.classic import munging_shared

        for cl in clients:
            cl.npf_consumption_on()

        clients = [munging_shared.LegacySimulatingClient.from_rate_limit_client(cl) for cl in clients]

    return clients
