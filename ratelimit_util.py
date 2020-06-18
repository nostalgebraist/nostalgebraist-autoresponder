"""a not-especially-good wrapper around the ratelimit info in tumblr API response"""
import pytumblr

class HeaderTumblrRequest(pytumblr.TumblrRequest):
    def json_parse(self, response):
        return response

class RateLimitClient(pytumblr.TumblrRestClient):
    def __init__(self, consumer_key, consumer_secret="", oauth_token="", oauth_secret="", host="https://api.tumblr.com", blogName=None):
        if blogName is None:
            # want to keep arg order the same as TumblrRestClient so blogName must be kwarg
            raise ValueError("must specify blogName")
        self.blogName = blogName

        self.request = HeaderTumblrRequest(consumer_key, consumer_secret, oauth_token, oauth_secret, host)


    @staticmethod
    def from_tumblr_rest_client(client: pytumblr.TumblrRestClient, blogName):
        return RateLimitClient(consumer_key=client.request.consumer_key,
                               consumer_secret=client.request.oauth.client.client_secret,
                               oauth_token=client.request.oauth.client.resource_owner_key,
                               oauth_secret=client.request.oauth.client.resource_owner_secret,
                               blogName=blogName
                               )

    def get_ratelimit_data(self):
        response = self.posts(self.blogName, limit=1)

        headers = response.headers
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
