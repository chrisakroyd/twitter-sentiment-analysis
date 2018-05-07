import { CHANGE_DASH_VIEW } from '../constants/actions';

export function changeDashView(view) {
  return {
    type: CHANGE_DASH_VIEW,
    view,
  };
}
