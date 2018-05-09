import React from 'react';
import PropTypes from 'prop-types';
import './architecture.scss';

function middleLayer(arch) {
  if (arch.concPool) {
    return (
      <div className="conc-pool">
      </div>
    );
  }
  return (
    <div className="attention">
    </div>
  );
}


const Architecture = ({ architecture }) =>
  (
    <div className="data-tile tile">
      <div className="embedding">
      </div>
      <i className="down-arrow material-icons">play_arrow</i>
      <div className="rnn-layers">
      </div>
      <i className="down-arrow material-icons">play_arrow</i>
      {
        middleLayer(architecture)
      }
      <i className="down-arrow material-icons">play_arrow</i>
      <div className="output">

      </div>
    </div>
  );

Architecture.propTypes = {
  architecture: PropTypes.shape({
    embeddingDim: PropTypes.number.isRequired,
    rnnUnits: PropTypes.number.isRequired,
    rnnLayers: PropTypes.number.isRequired,
    biDirectional: PropTypes.bool.isRequired,
    rnnParams: {
      l2: PropTypes.number.isRequired,
      dropout: PropTypes.number.isRequired,
    },
    attention: PropTypes.bool.isRequired,
    concPool: PropTypes.bool.isRequired,
    topK: PropTypes.number.isRequired,
    classes: PropTypes.number.isRequired,
  }).isRequired,
};

