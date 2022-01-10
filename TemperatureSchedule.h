#pragma once

class TemperatureSchedule {
public:
	virtual ~TemperatureSchedule() = default;
	virtual float temperature(float time) const = 0;
};
