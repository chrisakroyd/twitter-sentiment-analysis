import urlJoin from 'url-join';

import {
  SIDEBAR_LIVE, SIDEBAR_LIVE_LABEL,
  SIDEBAR_MODELS, SIDEBAR_MODELS_LABEL,
  SIDEBAR_DATA, SIDEBAR_DATA_LABEL,
  SIDEBAR_TRAIN, SIDEBAR_TRAIN_LABEL,
  SIDEBAR_RESULTS, SIDEBAR_RESULTS_LABEL,
} from '../../constants/sidebar';

function genSidebarItems(match) {
  return [
    {
      label: SIDEBAR_LIVE_LABEL,
      type: SIDEBAR_LIVE,
      icon: 'play_arrow',
      linkTo: match.url,
    },
    {
      label: SIDEBAR_DATA_LABEL,
      type: SIDEBAR_DATA,
      icon: 'donut_large',
      linkTo: urlJoin(match.url, '/data'),
    },
    {
      label: SIDEBAR_MODELS_LABEL,
      type: SIDEBAR_MODELS,
      icon: 'work',
      linkTo: urlJoin(match.url, '/models'),
    },
    {
      label: SIDEBAR_TRAIN_LABEL,
      type: SIDEBAR_TRAIN,
      icon: 'directions_run',
      linkTo: urlJoin(match.url, '/train'),
    },
    {
      label: SIDEBAR_RESULTS_LABEL,
      type: SIDEBAR_RESULTS,
      icon: 'show_chart',
      linkTo: urlJoin(match.url, '/results'),
    },
  ];
}

export default genSidebarItems;
