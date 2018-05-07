import { STATUS, STATUS_SUCCESS, STATUS_FAILURE } from '../constants/actions';

const status = (state = {}, action) => {
  switch (action.type) {
    case STATUS:
      return {
        status: {},
        loading: true,
      };
    case STATUS_SUCCESS:
      return {
        status: action.status,
        loading: false,
      };
    case STATUS_FAILURE:
      return {
        status: state.status,
        loading: false,
      };
    default:
      return state;
  }
};

export default status;
