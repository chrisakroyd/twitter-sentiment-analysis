import {
  DATASETS, DATASETS_SUCCESS, DATASETS_FAILURE,
  EMBEDDINGS, EMBEDDINGS_SUCCESS, EMBEDDINGS_FAILURE,
} from '../constants/actions';

const datasets = (state = {}, action) => {
  switch (action.type) {
    case DATASETS:
      return {
        datasets: [],
        loading: true,
      };
    case DATASETS_SUCCESS:
      return {
        datasets: action.data,
        loading: false,
      };
    case DATASETS_FAILURE:
      return {
        datasets: state.datasets,
        loading: false,
      };
    default:
      return state;
  }
};

const embeddings = (state = {}, action) => {
  switch (action.type) {
    case EMBEDDINGS:
      return {
        embeddings: [],
        loading: true,
      };
    case EMBEDDINGS_SUCCESS:
      return {
        embeddings: action.embeddings,
        loading: false,
      };
    case EMBEDDINGS_FAILURE:
      return {
        embeddings: state.embeddings,
        loading: false,
      };
    default:
      return state;
  }
};

export { datasets, embeddings };
