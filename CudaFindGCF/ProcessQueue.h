#pragma once

#include <vector>
#include <thread>

#include <iostream>
#include <exception>
#include <iterator>

template<class T>
class ProcessQueue{
public:
	void addTask(T param){
		if (!isSetup || isFinished) stateError();
		processes.push_back(param);
	}

	void clearQueue(){

		if (!isSetup || isFinished) stateError();


		for (size_t i = 0; i < processes.size(); i++){
			processSingle(processes[i]);
		}

		processes.clear();

	}

	void addSetup(void(*setup)()){
		if (isSetup || isFinished) stateError();
		setupTasks.push_back(setup);
	}

	void addProcess(void(*process)(T)){
		if (isFinished) stateError();
		processTasks.push_back(process);
	}
	void addCleanup(void(*cleanup)()){
		if (isFinished) stateError();
		cleanupTasks.push_back(cleanup);
	}
	void setup(){
		if (isSetup || isFinished) stateError();

		for (size_t i = 0; i < setupTasks.size(); i++){
			setupTasks[i]();
		}

		isSetup = true;
	}
	void finish(){
		if (!isSetup || isFinished) stateError();

		clearQueue();

		for (size_t i = 0; i < cleanupTasks.size(); i++){
			cleanupTasks[i]();
		}

		isFinished = true;
	}

private:
	std::vector<void(*)()> setupTasks;
	std::vector<void(*)(T)> processTasks;
	std::vector<void(*)()> cleanupTasks;
	
	void processSingle(T par){
		for (size_t i = 0; i < processTasks.size(); i++){
			processTasks[i](par);
		}
	}

	bool isSetup = false;
	bool isFinished = false;

	void stateError(){
		throw std::runtime_error("Class is in an illegal state");
	}
	std::vector<T> processes;
    
};

