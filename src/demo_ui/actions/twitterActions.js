import { TWEETS, TWEETS_SUCCESS, TWEETS_FAILURE } from '../constants/actions';

export function tweets() {
  return {
    type: TWEETS,
  };
}

export function tweetsSuccess(data) {
  return {
    type: TWEETS_SUCCESS,
    tweets: data,
  };
}

export function tweetsFailure(errorCode) {
  return {
    type: TWEETS_FAILURE,
    errorCode,
  };
}
