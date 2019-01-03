import React from 'react';
import PropTypes from 'prop-types';
import classNames from 'classnames';

import './demo.scss';

import InputBar from './InputBar/InputBar';
import SearchButton from './SearchButton/SearchButton';
import AnnotatedHighlight from './AnnotatedHighlight/AnnotatedHighlight';
import WordHeat from './WordHeat/WordHeat';
import ConfidenceGauge from './ConfidenceGauge/ConfidenceGauge';

import tweetShape from '../prop-shapes/tweetShape';

const Demo = ({ process, setText, activeText, tweets }) => {
  const textLabel = activeText.classification.toLowerCase();
  const classificationClass = classNames({ positive: textLabel === 'positive', negative: textLabel === 'negative' });

  return (
    <div className="live dash-body">
      <div className="body-header">
        <h1>Demo</h1>
      </div>

      <div className="tile-row">
        <div className="tile large-tile">
          <div className="tile-header">
            <h3>Demo</h3>
          </div>
          <div className="tile-body">
            <h4>1. Enter Text</h4>
            <div className="enter-text-row">
              <InputBar
                onEnter={process}
                value={activeText.text}
                onKeyPress={setText}
              />
              <div>
                <SearchButton onEnter={process}/>
              </div>
            </div>
            <div className="text-block">
              <h4>2. Processed Text</h4>
              <p>Before being run through the neural network, the text entered above is processed,
                annotated and tokenized. Annotated words are displayed in
                <div className="colour-word">this colour</div>.
              </p>
              <AnnotatedHighlight words={activeText.processed}/>
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
                words={activeText.processed}
                scores={activeText.attentionWeights}
              />
              <div className="classification-container">
                <div>
                  <span>Classification: </span>
                  <span
                    className={classificationClass}
                  >
                    {activeText.classification}
                  </span>
                </div>
                <ConfidenceGauge confidence={activeText.confidence}/>
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
  status: PropTypes.shape({
    memoryUsage: PropTypes.number.isRequired,
    maxMemoryUsage: PropTypes.number.isRequired,
    load: PropTypes.number.isRequired,
    graphicsCard: PropTypes.string.isRequired,
    connected: PropTypes.bool.isRequired,
  }).isRequired,
  activeText: PropTypes.shape({
    text: PropTypes.string.isRequired,
    processed: PropTypes.string.isRequired,
    attentionWeights: PropTypes.arrayOf(PropTypes.number.isRequired).isRequired,
    classification: PropTypes.string.isRequired,
    confidence: PropTypes.number.isRequired,
  }).isRequired,
  tweets: PropTypes.shape({
    tweets: PropTypes.arrayOf(tweetShape).isRequired,
  }).isRequired,
};

export default Demo;
