import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './demo.scss';

import WordHeat from './WordHeat/WordHeat';
import Donut from './Donut/Donut';

import Button from './common/Button';
import InputBar from './common/InputBar';

const Demo = ({ process, toggleToken, setText, predictions, text }) => {
  const textLabel = predictions.label.toLowerCase();
  const classificationClass = classNames('label-header', {
    positive: textLabel === 'positive',
    negative: textLabel === 'negative',
    neutral: textLabel === 'neutral',
  });

  return (
    <div className="dash-body">
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
                value={text.text}
                placeholder="Enter text here"
                onKeyPress={setText}
              />
              <Button onClick={process} label="Predict" />
            </div>
            <div className="text-block">
              <h4>2. Attention and Classification</h4>
              <p>
                Attention is a technique which focuses on the most pertinent information
                within the input and calculates a per-word relevance score. Below is a
                visualisation of the scores for the text input as well as its classification
                and our confidence in it.
                The strength of the colour reflects the strength of its impact.
              </p>
              <WordHeat
                onClick={toggleToken}
                tokens={predictions.tokens}
                enabled={predictions.enabled}
                scores={predictions.attentionWeights}
              />
              <div className="classification-container">
                <Donut probs={predictions.probs} />
                <div className="label-container">
                  <h3>We think this text is...</h3>
                  <h2 className={classificationClass}>{predictions.label}</h2>
                </div>
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
  toggleToken: PropTypes.func.isRequired,
  setText: PropTypes.func.isRequired,
  // Data
  predictions: PropTypes.shape({
    tokens: PropTypes.arrayOf(PropTypes.string).isRequired,
    enabled: PropTypes.arrayOf(PropTypes.bool).isRequired,
    attentionWeights: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
    label: PropTypes.string.isRequired,
    probs: PropTypes.arrayOf(PropTypes.number).isRequired,
  }).isRequired,
  text: PropTypes.shape({
    text: PropTypes.string.isRequired,
  }).isRequired,
};

export default Demo;
