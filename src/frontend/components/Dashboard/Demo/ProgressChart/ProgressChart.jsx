import React from 'react';
import * as d3 from 'd3';
import { withFauxDOM } from 'react-faux-dom';

import './progress-chart.scss';
import PropTypes from "prop-types";

const width = 130;
const height = 130;
const animationDuration = 750;

class ProgressChart extends React.Component {
  componentDidMount() {
    const faux = this.props.connectFauxDOM('div', 'chart');
    const ratio = this.props.value / this.props.max;
    const innerRadius = 65;

    const color = d3.scaleOrdinal().range(['#67BAF5', '#3eb2e3']);

    const arcLine = d3.arc()
      .innerRadius(innerRadius - 13)
      .outerRadius(innerRadius)
      .startAngle(0);

    const backgroundArc = d3.arc()
      .innerRadius(innerRadius - 13)
      .outerRadius(innerRadius)
      .startAngle(0)
      .endAngle(2 * Math.PI);

    const svg = d3.select(faux)
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .attr('class', 'shadow')
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`);

    const backgroundLine = svg.append('path')
      .attr('d', backgroundArc)
      .attr('fill', '#DEDEDE');

    const pathLine = svg.append('path')
      .datum({ endAngle: 0 })
      .attr('d', arcLine)
      .style('fill', color('Success'));

    const middleCount = svg.append('text')
      .text(d => `0.0 ${this.props.unit}`)
      .attr('class', 'middleText')
      .attr('text-anchor', 'middle')
      .attr('dy', 10)
      .style('fill', color('Success'))
      .style('font-size', '30px');

    const arcTween = (transition, newAngle) => {
      transition.attrTween('d', (d) => {
        const interpolate = d3.interpolate(d.endAngle, newAngle);
        const interpolateCount = d3.interpolate(0, this.props.value);
        return (t) => {
          d.endAngle = interpolate(t);
          middleCount.text(`${interpolateCount(t).toFixed(1)}${this.props.unit}`);
          return arcLine(d);
        };
      });
    };

    setTimeout(() => pathLine.transition()
      .duration(animationDuration)
      .call(arcTween, ((2 * Math.PI)) * ratio), 100);

    this.props.animateFauxDOM(animationDuration);
  }

  render() {
    return (
      <div className="progress-chart">
        {this.props.chart}
        <p>{this.props.label}</p>
      </div>
    );
  }
}

ProgressChart.propTypes = {
  label: PropTypes.string.isRequired,
  unit: PropTypes.string.isRequired,
  value: PropTypes.number.isRequired,
  max: PropTypes.number.isRequired,
};

export default withFauxDOM(ProgressChart);
