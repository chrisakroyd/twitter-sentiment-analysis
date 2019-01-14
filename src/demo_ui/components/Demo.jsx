import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './demo.scss';

import InputBar from './InputBar/InputBar';
import SearchButton from './SearchButton/SearchButton';
import WordHeat from './WordHeat/WordHeat';
import ConfidenceGauge from './ConfidenceGauge/ConfidenceGauge';


const Demo = ({ process, setText, predictions }) => {
  const textLabel = predictions.label.toLowerCase();
  const classificationClass = classNames({
    positive: textLabel === 'positive',
    negative: textLabel === 'negative',
    neutral: textLabel === 'neutral',
  });

  return (
    <div className="live dash-body">
      <div className="body-header">
        <h1>Demo</h1>
      </div>

      <div className="tile-row">
        <div className="tile large-tile">
          <div className="tile-header">
            <h1>Demo</h1>
          </div>
          <div className="tile-body">
            <h4>1. Enter Text</h4>
            <div className="enter-text-row">
              <InputBar
                onEnter={process}
                value={predictions.text}
                onKeyPress={setText}
              />
              <div>
                <SearchButton onEnter={process} />
              </div>
            </div>
            <div className="text-block">
              <h4>3. Attention, Classification and Confidence</h4>
              <p>
                Attention is a technique which focuses on the most pertinent information
                within the input and calculates a per-word relevance score. Below is a
                visualisation of the scores for the text input as well as its classification
                and our confidence in it.
                The strength of the colour reflects the strength of its impact.
              </p>
              <WordHeat
                tokens={predictions.tokens}
                scores={predictions.attentionWeights}
              />
              <div className="classification-container">
                <div>
                  <span>Classification: </span>
                  <span
                    className={classificationClass}
                  >
                    {predictions.label}
                  </span>
                </div>
                <ConfidenceGauge probs={predictions.probs[0]} />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

Demo.propTypes = {
  // Functions
  process: PropTypes.func.isRequired,
  setText: PropTypes.func.isRequired,
  // Data
  predictions: PropTypes.shape({
    text: PropTypes.string.isRequired,
    tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
    attentionWeights: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
    label: PropTypes.string.isRequired,
    probs: PropTypes.arrayOf(PropTypes.number).isRequired,
  }).isRequired,
};

export default Demo;
