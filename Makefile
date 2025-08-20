CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -march=native -ffast-math -std=gnu++20 \
           -fopenmp -fno-math-errno -ffp-contract=fast

SRC_DIR = src
BUILD_DIR = build
TARGET = app

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS))

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compilación de cada .cpp a .o con su .vec de reportes
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -fopt-info-vec=$(@:.o=.vec) -c $< -o $@

.PHONY: clean run

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)