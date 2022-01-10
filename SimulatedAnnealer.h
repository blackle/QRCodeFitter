#pragma once
#include <random>
#include <cstdlib>
#include "TemperatureSchedule.h"

template<typename T>
class NeighbourGenerator;

template<typename T>
class EnergyFunction;

template<typename T>
class SimulatedAnnealer {
public:
	SimulatedAnnealer(int numIters, const T& initialState, const NeighbourGenerator<T>* generator, const EnergyFunction<T>* energy, const TemperatureSchedule* schedule);
	~SimulatedAnnealer();

	void anneal();

	const T& currentState() const;
	float currentEnergy() const;

private:
	int _numIters;
	T _currentState;
	float _currEnergy;
	const NeighbourGenerator<T>* _generator;
	const EnergyFunction<T>* _energy;
	const TemperatureSchedule* _schedule;
};

template<typename T>
SimulatedAnnealer<T>::SimulatedAnnealer(int numIters, const T& initialState, const NeighbourGenerator<T>* generator, const EnergyFunction<T>* energy, const TemperatureSchedule* schedule)
	: _numIters(numIters)
	, _currentState(initialState)
	, _generator(generator)
	, _energy(energy)
	, _schedule(schedule)
{
	_currEnergy = _energy->energy(_currentState);
}

template<typename T>
SimulatedAnnealer<T>::~SimulatedAnnealer() {}

template<typename T>
void SimulatedAnnealer<T>::anneal() {
	for (int i = 0; i < _numIters; i++) {
		float time = (float)(i) / (float)(_numIters - 1);
		float temp = _schedule->temperature(time);

		auto candidate = _generator->generate(_currentState);

		float candidateEnergy = _energy->energy(candidate);
		float diff = candidateEnergy - _currEnergy;

		float acceptanceProb = (candidateEnergy < _currEnergy) ? 1.0 : std::exp(-diff / temp);

		if (i % 100 == 0) {
			std::cout << "currEnergy: " << _currEnergy << std::endl;
		}

		std::random_device device;
		std::mt19937 generator(device());
		std::bernoulli_distribution dist(acceptanceProb);

		if (dist(generator)) {
			_currentState = candidate;
			_currEnergy = candidateEnergy;
		}
	}
}

template<typename T>
const T& SimulatedAnnealer<T>::currentState() const {
	return _currentState;
}

template<typename T>
float SimulatedAnnealer<T>::currentEnergy() const {
	return _energy->energy(_currentState);
}
