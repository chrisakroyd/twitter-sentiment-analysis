import React from 'react';
import PropTypes from 'prop-types';

import { interpolate } from 'd3-interpolate';
import './confidence-gauge.scss';

const interpolateLin = interpolate(0.0, 100.0);

const ConfidenceGauge = ({ confidence }) => {
  const confidenceWidth = { width: `${interpolateLin(confidence)}%` };
  let level = 'Low';

  if (confidence > 0.33 && confidence < 0.66) {
    level = 'Medium';
  } else if (confidence > 0.66) {
    level = 'High';
  }

  return (
    <div className="confidence-gauge">
      <div className="confidence-level" style={confidenceWidth} />
      <span>{level} Confidence</span>
    </div>
  );
};

ConfidenceGauge.propTypes = {
  confidence: PropTypes.number.isRequired,
};

export default ConfidenceGauge;
