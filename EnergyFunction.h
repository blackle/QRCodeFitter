#pragma once

template<typename T>
class EnergyFunction {
public:
	virtual ~EnergyFunction() = default;
	virtual float energy(const T& state) const = 0;
};
