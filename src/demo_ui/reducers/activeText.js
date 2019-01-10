import { PREDICT, PREDICT_SUCCESS, PREDICT_FAILURE, SET_INPUT_TEXT } from '../constants/actions';

const search = (state = {}, action) => {
  switch (action.type) {
    case PREDICT:
      return Object.assign({}, state, {
        processed: '',
        attentionWeights: [],
        classification: '',
        confidence: 0.0,
        loading: true,
      });
    case PREDICT_SUCCESS:
      return Object.assign({}, state, {
        processed: action.data[0].processed,
        attentionWeights: action.data[0].attentionWeights,
        classification: action.data[0].classification,
        confidence: action.data[0].confidence,
        loading: false,
      });
    case PREDICT_FAILURE:
      return Object.assign({}, state, {
        text: action.text,
        loading: false,
        errorCode: action.error_code,
        errorMessage: action.error_message,
      });
    case SET_INPUT_TEXT:
      return Object.assign({}, state, {
        originalText: action.text,
        text: action.text,
        loading: false,
      });
    default:
      return state;
  }
};

export default search;
