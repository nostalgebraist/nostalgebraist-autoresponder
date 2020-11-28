"""a wrapper around pytumblr's client with tools for ratelimit info, no $&^!ing param validation, etc"""
import pytumblr

RAW_RESPONSES_FOR_DEBUG = False
LOG_CALLS_FOR_DEBUG = False

class HeaderTumblrRequest(pytumblr.TumblrRequest):
    def json_parse(self, response):
        self.last_headers = response.headers
        if RAW_RESPONSES_FOR_DEBUG:
        return response
        return super().json_parse(response)

class RateLimitClient(pytumblr.TumblrRestClient):
    def __init__(self, consumer_key, consumer_secret="", oauth_token="", oauth_secret="", host="https://api.tumblr.com", blogName=None):
        if blogName is None:
            # want to keep arg order the same as TumblrRestClient so blogName must be kwarg
            raise ValueError("must specify blogName")
        self.blogName = blogName

        self.request = HeaderTumblrRequest(consumer_key, consumer_secret, oauth_token, oauth_secret, host)

    def send_api_request(self, method, url, params={}, valid_parameters=[], needs_api_key=False):
        if LOG_CALLS_FOR_DEBUG:
            print(f"!requesting {url} with {repr(params)}")
        extras = [key for key in params.keys() if key not in valid_parameters]
        valid_parameters_extended = valid_parameters + extras
        return super().send_api_request(method,
                                        url,
                                        params=params,
                                        valid_parameters=valid_parameters_extended,
                                        needs_api_key=needs_api_key)

    @staticmethod
    def from_tumblr_rest_client(client: pytumblr.TumblrRestClient, blogName):
        return RateLimitClient(consumer_key=client.request.consumer_key,
                               consumer_secret=client.request.oauth.client.client_secret,
                               oauth_token=client.request.oauth.client.resource_owner_key,
                               oauth_secret=client.request.oauth.client.resource_owner_secret,
                               blogName=blogName
                               )

    def get_ratelimit_data(self):
        if not hasattr(self.request, "last_headers"):
            print("warning: no ratelimit data found, sending a request from ratelimit client")
        response = self.posts(self.blogName, limit=1)

        headers = self.request.last_headers
        results = {}

        results["day"] = {"remaining": int(headers["X-Ratelimit-Perday-Remaining"]),
                          "reset": int(headers["X-Ratelimit-Perday-Reset"])}

        results["hour"] = {"remaining": int(headers["X-Ratelimit-Perhour-Remaining"]),
                           "reset": int(headers["X-Ratelimit-Perhour-Reset"])}


        for k in ["day", "hour"]:
            results[k]["max_rate"] = results[k]["remaining"] / results[k]["reset"]

        results["effective_max_rate"] = min([results[k]["max_rate"] for k in ["day", "hour"]])
        results["effective_remaining"] = min([results[k]["remaining"] for k in ["day", "hour"]])
        results["effective_reset"] = min([results[k]["reset"] for k in ["day", "hour"]])

        return results
