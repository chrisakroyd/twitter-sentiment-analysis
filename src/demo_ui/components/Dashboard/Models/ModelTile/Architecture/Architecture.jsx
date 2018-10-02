import React from 'react';
import PropTypes from 'prop-types';
import './architecture.scss';


class Architecture extends React.Component {
  rnnLayers(arch) {
    const layers = [];
    for (let i = 0; i < arch.rnnLayers; i+= 1) {
      layers.push(
        (
          <div className="rnn-layer">
            <div className="arch-block forward">{arch.rnnUnits} LSTM - Fwd</div>
            <div className="arch-block backward">{arch.rnnUnits} LSTM - Bck</div>
            <i className="down-arrow material-icons">play_arrow</i>
          </div>
        )
      );
    }

    return layers;
  }

  middleLayer(arch) {
    if (arch.concPool) {
      return (
        <div className="conc-pool-container">
          <div className="vec-container">
            <div className="arch-block vec-block">
              Avg
            </div>
            <div className="arch-block vec-block">
              Max
            </div>
            <div className="arch-block vec-block">
              Last
            </div>
            <div className="arch-block top-k">
              Top-{arch.topK}
            </div>
          </div>
          <i className="down-arrow material-icons">play_arrow</i>
          <div className="arch-block conc-pool">
            {((arch.rnnUnits * 2) * 3) + ((arch.rnnUnits * 2) * arch.topK)} Length Feature Vector
          </div>
        </div>
      );
    }
    return (
      <div className="arch-block attention">
        {arch.rnnUnits * 2} Attention Units
      </div>
    );
  }

  render() {
    const architecture = this.props.architecture;

    return (
      <div className="architecture">
        <p>Tweet</p>
        <i className="down-arrow material-icons">play_arrow</i>
        <div className="arch-block embedding">Embedding - {architecture.embeddingDim}d</div>
        <i className="down-arrow material-icons">play_arrow</i>
        <div className="rnn-layers">
          {this.rnnLayers(architecture)}
          <div className="concat arch-block">
            <div className="concat-block">{architecture.rnnUnits} Forward</div>
            <div className="concat-block">{architecture.rnnUnits} Backward</div>
          </div>
        </div>
        <i className="down-arrow material-icons">play_arrow</i>
        {
          this.middleLayer(architecture)
        }
        <i className="down-arrow material-icons">play_arrow</i>
        <div className="output">
          <div className="arch-block out-cell positive"></div>
          <div className="arch-block out-cell neutral"></div>
          <div className="arch-block out-cell negative"></div>
        </div>
      </div>
    );
  }
}

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

export default Architecture;
