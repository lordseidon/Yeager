#include "remote_evaluator.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>  
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL REMOTE_EVALUATOR_ARRAY_API
#include <numpy/arrayobject.h>

// Helper function to initialize NumPy - returns int to match macro expectations
static int initialize_numpy() {
    import_array1(-1);
    return 0;
}

RemoteEvaluator::RemoteEvaluator(const std::string& model_path, int batch_size, int min_batch_size)
    : batch_size_(batch_size), min_batch_size_(min_batch_size), stop_thread_(false), 
      main_thread_state_(nullptr), python_evaluator_(nullptr), device_worker_(nullptr) {
    
    // Initialize Python interpreter if not already done
    if (!Py_IsInitialized()) {
        Py_Initialize();
        PyEval_InitThreads();
        if (initialize_numpy() < 0) {
            throw std::runtime_error("Failed to initialize NumPy");
        }
    }
    
    // Initialize Python model with GIL held
    {
        PyGILState_STATE gstate = PyGILState_Ensure();
        initialize_python_model(model_path);
        PyGILState_Release(gstate);
    }
    
    // Save the main thread state and release GIL so worker thread can use it
    main_thread_state_ = PyEval_SaveThread();
    
    // Start the processing thread
    processing_thread_ = std::thread(&RemoteEvaluator::processing_loop, this);
}

RemoteEvaluator::~RemoteEvaluator() {
    stop_thread_ = true;
    cv_.notify_all();
    
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
    
    // Clean up Python objects
    if (Py_IsInitialized()) {
        PyGILState_STATE gstate = PyGILState_Ensure();
        Py_XDECREF(device_worker_);
        Py_XDECREF(python_evaluator_);
        PyGILState_Release(gstate);
    }
}

void RemoteEvaluator::initialize_python_model(const std::string& model_path) {
    // Add current directory and server directory to Python path
    PyObject* sys = PyImport_ImportModule("sys");
    PyAssert(sys);
    PyObject* sysPath = PyObject_GetAttrString(sys, "path");
    PyAssert(sysPath);
    
    PyObject* currentDir = PyUnicode_FromString(".");
    PyAssert(currentDir);
    PyList_Append(sysPath, currentDir);
    Py_DECREF(currentDir);
    
    PyObject* serverDir = PyUnicode_FromString("./server");
    PyAssert(serverDir);
    PyList_Append(sysPath, serverDir);
    Py_DECREF(serverDir);
    
    Py_DECREF(sysPath);
    Py_DECREF(sys);
    
    // Import the model module (remove .py extension if present)
    std::string module_name = model_path;
    if (module_name.size() > 3 && module_name.substr(module_name.size() - 3) == ".py") {
        module_name = module_name.substr(0, module_name.size() - 3);
    }
    
    PyObject* model_module = PyImport_ImportModule(module_name.c_str());
    PyAssert(model_module);
    
    // Get the ChessEvaluator class
    PyObject* evaluator_class = PyObject_GetAttrString(model_module, "ChessEvaluator");
    PyAssert(evaluator_class);
    PyAssert(PyCallable_Check(evaluator_class));
    
    // Instantiate the evaluator with model path and num_tpu_devices=8
    PyObject* args = PyTuple_New(2);
    PyObject* model_path_str = PyUnicode_FromString("/mnt/disk/yeager/model_analysis/models/alphazero_model.h5");
    PyObject* num_devices = PyLong_FromLong(8);
    PyTuple_SetItem(args, 0, model_path_str);
    PyTuple_SetItem(args, 1, num_devices);
    
    python_evaluator_ = PyObject_CallObject(evaluator_class, args);
    PyAssert(python_evaluator_);
    Py_DECREF(args);
    
    // Get a device worker (device 0) and its predict method
    PyObject* get_device_worker = PyObject_GetAttrString(python_evaluator_, "get_device_worker");
    PyAssert(get_device_worker);
    PyAssert(PyCallable_Check(get_device_worker));
    
    PyObject* device_id = PyLong_FromLong(0);
    device_worker_ = PyObject_CallFunctionObjArgs(get_device_worker, device_id, nullptr);
    PyAssert(device_worker_);
    Py_DECREF(device_id);
    Py_DECREF(get_device_worker);
    
    Py_DECREF(evaluator_class);
    Py_DECREF(model_module);
}

std::future<EvaluationResult> RemoteEvaluator::queue_request(std::vector<float>&& position_tensor) {
    InferenceRequest req;
    req.position_tensor = std::move(position_tensor);
    auto future = req.promise.get_future();
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        request_queue_.push_back(std::move(req));
    }
    cv_.notify_one();
    
    return future;
}

void RemoteEvaluator::processing_loop() {
    // Ensure NumPy is initialized in this thread
    PyGILState_STATE init_gstate = PyGILState_Ensure();
    if (initialize_numpy() < 0) {
        std::cerr << "[RemoteEvaluator] Failed to initialize NumPy in worker thread" << std::endl;
        PyGILState_Release(init_gstate);
        return;
    }
    PyGILState_Release(init_gstate);
    
    last_batch_time_ = std::chrono::steady_clock::now();
    
    while (!stop_thread_.load()) {
        std::deque<InferenceRequest> batch;
        
        auto wait_start = std::chrono::high_resolution_clock::now();
        
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            
            // Wait with timeout for minimum batch size
            auto now = std::chrono::steady_clock::now();
            auto time_since_last_batch = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - last_batch_time_).count();
            
            cv_.wait_for(lock, std::chrono::milliseconds(BATCH_TIMEOUT_MS), [this, time_since_last_batch] {
                return stop_thread_.load() || 
                       request_queue_.size() >= static_cast<size_t>(min_batch_size_) ||
                       (!request_queue_.empty() && time_since_last_batch >= BATCH_TIMEOUT_MS);
            });
            
            if (stop_thread_.load() && request_queue_.empty()) {
                break;
            }
            
            // Collect up to 256 requests
            while (!request_queue_.empty() && batch.size() < 256) {
                batch.push_back(std::move(request_queue_.front()));
                request_queue_.pop_front();
            }
        }
        
        auto wait_end = std::chrono::high_resolution_clock::now();
        auto wait_time = std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();
        
        if (batch.empty()) {
            continue;
        }
        
        last_batch_time_ = std::chrono::steady_clock::now();
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        try {
            // TIMING: Prepare batch data
            auto prep_start = std::chrono::high_resolution_clock::now();
            
            const size_t tensor_size = 8 * 8 * 108;
            const size_t FIXED_BATCH_SIZE = 256;
            
            // Store actual number of valid requests
            size_t num_valid_requests = batch.size();
            
            // Always create batch of 256 (pad with zeros)
            std::vector<float> batch_data(FIXED_BATCH_SIZE * tensor_size, 0.0f);
            
            size_t valid_requests = 0;
            for (const auto& req : batch) {
                if (req.position_tensor.size() == tensor_size) {
                    std::copy(req.position_tensor.begin(), 
                             req.position_tensor.end(),
                             batch_data.begin() + valid_requests * tensor_size);
                    valid_requests++;
                } else {
                    std::cerr << "[RemoteEvaluator] Skipping invalid tensor size: " 
                              << req.position_tensor.size() << std::endl;
                }
            }
            
            auto prep_end = std::chrono::high_resolution_clock::now();
            auto prep_time = std::chrono::duration_cast<std::chrono::microseconds>(prep_end - prep_start).count();
            
            if (valid_requests == 0) {
                std::cerr << "[RemoteEvaluator] No valid requests in batch" << std::endl;
                continue;
            }
            
            // TIMING: Acquire GIL
            auto gil_start = std::chrono::high_resolution_clock::now();
            PyGILState_STATE gstate = PyGILState_Ensure();
            auto gil_end = std::chrono::high_resolution_clock::now();
            auto gil_time = std::chrono::duration_cast<std::chrono::microseconds>(gil_end - gil_start).count();
            
            // TIMING: Create numpy array (always 256, 8, 8, 108)
            auto numpy_start = std::chrono::high_resolution_clock::now();
            
            npy_intp dims[4] = {
                static_cast<npy_intp>(FIXED_BATCH_SIZE),
                static_cast<npy_intp>(8),
                static_cast<npy_intp>(8),
                static_cast<npy_intp>(108)
            };
            
            PyObject* np_batch = PyArray_SimpleNew(4, dims, NPY_FLOAT32);
            if (!np_batch) {
                PyErr_Print();
                PyGILState_Release(gstate);
                throw std::runtime_error("Failed to create numpy array");
            }
            
            float* np_data = static_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(np_batch)));
            std::copy(batch_data.begin(), batch_data.end(), np_data);
            
            auto numpy_end = std::chrono::high_resolution_clock::now();
            auto numpy_time = std::chrono::duration_cast<std::chrono::microseconds>(numpy_end - numpy_start).count();
            
            // TIMING: Call worker.predict(batch)
            auto python_call_start = std::chrono::high_resolution_clock::now();
            
            PyObject* predict_method = PyObject_GetAttrString(device_worker_, "predict");
            if (!predict_method || !PyCallable_Check(predict_method)) {
                Py_XDECREF(predict_method);
                Py_DECREF(np_batch);
                PyGILState_Release(gstate);
                throw std::runtime_error("predict method not found or not callable");
            }
            
            PyObject* result = PyObject_CallFunctionObjArgs(predict_method, np_batch, nullptr);
            Py_DECREF(predict_method);
            
            auto python_call_end = std::chrono::high_resolution_clock::now();
            auto python_call_time = std::chrono::duration_cast<std::chrono::microseconds>(python_call_end - python_call_start).count();
            
            // We're done with the input array
            Py_DECREF(np_batch);
            
            if (!result) {
                std::cerr << "[RemoteEvaluator] Python function call failed:" << std::endl;
                PyErr_Print();
                PyGILState_Release(gstate);
                throw std::runtime_error("Python predict call failed");
            }
            
            // TIMING: Extract results
            auto extract_start = std::chrono::high_resolution_clock::now();
            
            if (!PyDict_Check(result)) {
                Py_DECREF(result);
                PyGILState_Release(gstate);
                throw std::runtime_error("predict did not return a dictionary");
            }
            
            PyObject* policies_obj = PyDict_GetItemString(result, "policies");
            PyObject* values_obj = PyDict_GetItemString(result, "values");
            
            if (!policies_obj || !values_obj) {
                Py_DECREF(result);
                PyGILState_Release(gstate);
                throw std::runtime_error("Result dictionary missing 'policies' or 'values'");
            }
            
            if (!PyArray_Check(policies_obj) || !PyArray_Check(values_obj)) {
                Py_DECREF(result);
                PyGILState_Release(gstate);
                throw std::runtime_error("Policies or values are not numpy arrays");
            }
            
            PyArrayObject* policies = reinterpret_cast<PyArrayObject*>(policies_obj);
            PyArrayObject* values = reinterpret_cast<PyArrayObject*>(values_obj);
            
            npy_intp* policies_shape = PyArray_DIMS(policies);
            size_t policy_size = static_cast<size_t>(policies_shape[1]);
            
            // TIMING: Copy to C++ vectors (only valid_requests, ignore padding)
            auto copy_start = std::chrono::high_resolution_clock::now();
            
            std::vector<std::vector<float>> all_policies(valid_requests);
            std::vector<float> all_values(valid_requests);
            
            float* policies_ptr = static_cast<float*>(PyArray_DATA(policies));
            float* values_ptr = static_cast<float*>(PyArray_DATA(values));
            
            // Only copy the valid requests (ignore padded zeros)
            for (size_t i = 0; i < valid_requests; ++i) {
                all_policies[i].resize(policy_size);
                std::copy(policies_ptr + i * policy_size, 
                         policies_ptr + (i + 1) * policy_size,
                         all_policies[i].begin());
                all_values[i] = values_ptr[i];
            }
            
            auto copy_end = std::chrono::high_resolution_clock::now();
            auto copy_time = std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();
            
            auto extract_end = std::chrono::high_resolution_clock::now();
            auto extract_time = std::chrono::duration_cast<std::chrono::microseconds>(extract_end - extract_start).count();
            
            // Clean up Python objects
            Py_DECREF(result);
            
            // TIMING: Release GIL
            auto gil_release_start = std::chrono::high_resolution_clock::now();
            PyGILState_Release(gstate);
            auto gil_release_end = std::chrono::high_resolution_clock::now();
            auto gil_release_time = std::chrono::duration_cast<std::chrono::microseconds>(gil_release_end - gil_release_start).count();
            
            // TIMING: Distribute results
            auto distribute_start = std::chrono::high_resolution_clock::now();
            
            for (size_t i = 0; i < valid_requests; ++i) {
                EvaluationResult eval_result;
                eval_result.policy.assign(all_policies[i].begin(), all_policies[i].end());
                eval_result.value = all_values[i];
                batch[i].promise.set_value(std::move(eval_result));
            }
            
            auto distribute_end = std::chrono::high_resolution_clock::now();
            auto distribute_time = std::chrono::duration_cast<std::chrono::microseconds>(distribute_end - distribute_start).count();
            
            auto batch_end = std::chrono::high_resolution_clock::now();
            auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count();
            
            // Log every 2nd batch with detailed timing
            // static int batch_count = 0;
            // if (++batch_count % 10 == 0) {
            //     std::cout << "\n[Batch #" << batch_count << "] " << valid_requests 
            //               << " valid positions (padded to 256)" << std::endl;
            //     std::cout << "  Wait for batch:    " << std::setw(8) << wait_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  Prep data:         " << std::setw(8) << prep_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  Acquire GIL:       " << std::setw(8) << gil_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  Create numpy:      " << std::setw(8) << numpy_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  Python call:       " << std::setw(8) << python_call_time / 1000.0 << " ms  <-- NN inference" << std::endl;
            //     std::cout << "  Extract results:   " << std::setw(8) << extract_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "    (Copy to C++:    " << std::setw(8) << copy_time / 1000.0 << " ms)" << std::endl;
            //     std::cout << "  Release GIL:       " << std::setw(8) << gil_release_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  Distribute:        " << std::setw(8) << distribute_time / 1000.0 << " ms" << std::endl;
            //     std::cout << "  TOTAL:             " << std::setw(8) << total_time / 1000.0 << " ms" << std::endl;
            // }
            
        } catch (const std::exception& e) {
            std::cerr << "[RemoteEvaluator] Error: " << e.what() << std::endl;
            
            for (auto& req : batch) {
                try {
                    req.promise.set_exception(std::current_exception());
                } catch (...) {}
            }
        }
    }
}

void RemoteEvaluator::PyAssert(bool result) {
    if (!result) {
        PyErr_Print();
    }
    assert(result);
}