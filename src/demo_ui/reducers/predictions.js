import {
  PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE, PREDICT_TOKENS,
  PREDICT_TOKENS_SUCCESS, PREDICT_TOKENS_FAILURE, TOGGLE_TOKEN,
} from '../constants/actions';

const predictions = (state = {}, action) => {
  switch (action.type) {
    case PREDICT: {
      return Object.assign({}, state, {
        tokens: [],
        attentionWeights: [],
        label: '',
        probs: [],
        loading: true,
        error: null,
      });
    }
    case PREDICT_SUCCESS: {
      const prediction = action.data;

      return Object.assign({}, state, {
        tokens: prediction.tokens,
        enabled: new Array(prediction.tokens.length).fill(true),
        attentionWeights: prediction.attentionWeights,
        label: prediction.label,
        probs: prediction.probs,
        loading: false,
      });
    }
    case PREDICT_FAILURE || PREDICT_TOKENS_FAILURE: {
      return Object.assign({}, state, {
        loading: false,
        error: action.error,
      });
    }
    case PREDICT_TOKENS: {
      return Object.assign({}, state, {
        loading: true,
        error: null,
      });
    }
    case PREDICT_TOKENS_SUCCESS: {
      const prediction = action.data;
      const newWeights = [];
      let weightPointer = 0;

      for (let i = 0; i < state.enabled.length; i++) {
        if (state.enabled[i]) {
          newWeights.push(prediction.attentionWeights[weightPointer]);
          weightPointer += 1;
        } else {
          newWeights.push(0.0);
        }
      }

      return Object.assign({}, state, {
        attentionWeights: newWeights,
        label: prediction.label,
        probs: prediction.probs,
        loading: false,
      });
    }
    case TOGGLE_TOKEN: {
      const enabledTokens = state.enabled;
      enabledTokens[action.index] = !enabledTokens[action.index];
      return Object.assign({}, state, {
        enabled: enabledTokens,
      });
    }
    default: {
      return state;
    }
  }
};

export default predictions;
