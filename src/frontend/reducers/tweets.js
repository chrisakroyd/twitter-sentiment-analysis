import { TWEETS, TWEETS_SUCCESS, TWEETS_FAILURE } from '../constants/actions';

function preprocessTweets(rawTweets) {
  let tweets = [];
  if ('class' in rawTweets) {
    const ids = Object.keys(rawTweets.class);
    const text = rawTweets.text;
    const classes = rawTweets.class;
    tweets = ids.map(id => ({
      tweetId: id,
      username: 'UNKNWN',
      text: text[id],
      classification: classes[id],
    }));
  }
  return tweets;
}

const tweets = (state = {}, action) => {
  switch (action.type) {
    case TWEETS:
      return {
        tweets: [],
        loading: true,
      };
    case TWEETS_SUCCESS:
      return {
        tweets: preprocessTweets(action.tweets),
        loading: false,
      };
    case TWEETS_FAILURE:
      return {
        tweets: state.tweets,
        loading: false,
        errorCode: action.error_code,
        errorMessage: action.error_message,
      };
    default:
      return state;
  }
};

export default tweets;
