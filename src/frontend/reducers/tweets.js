import { TWEETS, TWEETS_SUCCESS, TWEETS_FAILURE } from '../constants/actions';

const tweets = (state = {}, action) => {
  switch (action.type) {
    case TWEETS:
      return {
        tweets: [],
        loading: true,
      };
    case TWEETS_SUCCESS:
      return {
        tweets: action.tweets,
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
