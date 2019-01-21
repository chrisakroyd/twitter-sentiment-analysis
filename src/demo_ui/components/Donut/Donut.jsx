import React from 'react';
import PropTypes from 'prop-types';
import shortid from 'shortid';
import * as d3 from 'd3';

import { classesShape } from '../../prop-shapes';

const DONUT_HEIGHT = 250;
const DONUT_WIDTH = 450;
const colourMap = {
  positive: '#5fcf80',
  neutral: 'gray',
  negative: '#FF6666',
};

const labelMap = {
  positive: 'Positive',
  neutral: 'Neutral',
  negative: 'Negative',
};


function midAngle(d) {
  return d.startAngle + (d.endAngle - d.startAngle) / 2;
}

class Donut extends React.Component {
  render() {
    const radius = Math.min(DONUT_WIDTH, DONUT_HEIGHT) / 2;
    const pie = d3.pie().sort(null).value(d => d);
    const arc = d3.arc().innerRadius(radius * 0.8).outerRadius(radius * 0.65);
    const outerArc = d3.arc()
      .outerRadius(radius * 0.9)
      .innerRadius(radius * 0.9);
    const test = pie(this.props.probs);
    const transform = `translate(${DONUT_WIDTH / 2}, ${DONUT_HEIGHT / 2})`;

    return (
      <svg ref={this.element} height={DONUT_HEIGHT} width={DONUT_WIDTH}>
        <g className="labels" transform={transform}>
          {
            test.map((prob, i) => {
              const pos = outerArc.centroid(prob);
              pos[0] = radius * 0.95 * (midAngle(prob) < Math.PI ? 1 : -1);
              const anchor = (midAngle(prob)) < Math.PI ? 'start' : 'end';
              return (
                <text
                  key={shortid.generate()}
                  dy=".35em"
                  textAnchor={anchor}
                  stroke={colourMap[this.props.classes[i]]}
                  transform={`translate(${pos})`}
                >
                  {labelMap[this.props.classes[i]]} {(prob.data * 100).toFixed(1)}%
                </text>);
            })
          }
        </g>
        <g className="lines" transform={transform}>
          {
            test.map((prob, i) => {
              const pos = outerArc.centroid(prob);
              pos[0] = radius * 0.95 * (midAngle(prob) < Math.PI ? 1 : -1);
              const test2 = [arc.centroid(prob), outerArc.centroid(prob), pos];
              return <polyline key={shortid.generate()} points={test2} stroke={colourMap[this.props.classes[i]]} />;
            })
          }
        </g>
        <g className="arcs" transform={transform}>
          {
            test.map((prob, i) => (
              <path
                key={shortid.generate()}
                d={arc(prob)}
                fill={colourMap[this.props.classes[i]]}
              />))
          }
        </g>
      </svg>
    );
  }
}

Donut.propTypes = {
  probs: PropTypes.arrayOf(PropTypes.number).isRequired,
  classes: classesShape.isRequired,
};

export default Donut;
