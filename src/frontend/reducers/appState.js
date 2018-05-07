import { CHANGE_DASH_VIEW } from '../constants/actions';

const appState = (state = {}, action) => {
  switch (action.type) {
    case CHANGE_DASH_VIEW:
      return {
        activeView: action.view,
      };
    default:
      return state;
  }
};

export default appState;
