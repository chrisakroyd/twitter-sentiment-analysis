import { MODELS, MODELS_SUCCESS, MODELS_FAILURE } from '../constants/actions';

const models = (state = {}, action) => {
  switch (action.type) {
    case MODELS:
      return {
        models: [],
        loading: true,
      };
    case MODELS_SUCCESS:
      return {
        models: action.models,
        loading: false,
      };
    case MODELS_FAILURE:
      return {
        models: state.models,
        loading: false,
      };
    default:
      return state;
  }
};

export default models;
