import { NEURAL, NEURAL_SUCCESS, NEURAL_FAILURE, SET_INPUT_TEXT } from '../constants/actions';

const search = (state = {}, action) => {
  switch (action.type) {
    case NEURAL:
      return {
        originalText: state.text,
        text: state.text,
        processed: '',
        attentionWeights: [],
        classification: '',
        confidence: 0.0,
        loading: true,
      };
    case NEURAL_SUCCESS:
      return {
        originalText: action.text,
        text: action.text,
        processed: action.processed,
        attentionWeights: action.attentionWeights,
        classification: action.classification,
        confidence: action.confidence,
        loading: false,
      };
    case NEURAL_FAILURE:
      return {
        originalText: action.text,
        text: action.text,
        processed: state.processed,
        attentionWeights: state.attentionWeights,
        classification: state.classification,
        confidence: state.confidence,
        loading: false,
        errorCode: action.error_code,
        errorMessage: action.error_message,
      };
    case SET_INPUT_TEXT:
      return {
        originalText: action.text,
        text: action.text,
        processed: state.processed,
        attentionWeights: state.attentionWeights,
        classification: state.classification,
        confidence: state.confidence,
        loading: false,
      };
    default:
      return state;
  }
};

export default search;
