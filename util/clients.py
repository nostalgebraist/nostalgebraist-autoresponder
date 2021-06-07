from bot_config import BotSpecificConstants


def get_me_clients(npf=True):
    bot_specific_constants = BotSpecificConstants.load()

    clients = bot_specific_constants.private_clients + bot_specific_constants.dashboard_clients

    if npf:
        for cl in clients:
            cl.npf_consumption_on()

    return clients
