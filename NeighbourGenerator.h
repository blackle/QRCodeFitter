#pragma once

template<typename T>
class NeighbourGenerator {
public:
	virtual ~NeighbourGenerator() = default;
	virtual const T generate(const T& state) const = 0;
};
