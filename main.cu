#include <iostream>
#include <chrono>
#include <iomanip>
#include "SDL2/SDL.h"
#include "cuda.h"
#include "CAMPARY/Doubles/src_gpu/multi_prec.h"

#define COORD multi_prec<2>

const int WIDTH = 640 * 2; //640
const int HEIGHT = 480 * 2; //480
//const double startX = -1.3733926023949114;
//const double startY = -0.08556614599092829;
//const double startZoom = 500000;

const COORD startX = -1.95379887674656838037;
const COORD startY = -0.00000000160728156195;
const COORD startZoom = 2.13951e+17;
const int MOVE_AMOUNT = 5;

using namespace std::chrono;

struct HSV {
  float h;
  float s;
  float v;
};

struct RGB {
  int r;
  int g;
  int b;
};

__device__
RGB Hsv2rgb(const HSV& color) {
  float s = color.s / 100;
  float v = color.v / 100;
  float c = v * s;
  float h = color.h / 60;
  float x = c * (1 - abs(fmod(h, (float)2) - 1));
  float m = v - c;
  int hi = (int) floor(h) % 6;
  HSV converted;
  switch(hi) {
    case 0: converted = {c, x, 0};
      break;
    case 1: converted = {x, c, 0};
      break;
    case 2: converted = {0, c, x};
      break;
    case 3: converted = {0, x, c};
      break;
    case 4: converted = {x, 0, c};
      break;
    case 5: converted = {c, 0, x};
      break;
    default: return {0, 0, 0};
  }
  int r = (converted.h + m) * 255;
  int g = (converted.s + m) * 255;
  int b = (converted.v + m) * 255;
  return {r, g, b};
}

void Draw(SDL_Renderer * ren, SDL_Texture * texture, int * data) {
  // SDL_SetRenderDrawColor(ren, 0, 0, 0, 255);
  // SDL_RenderClear(ren);
  // for(int x = 0; x < WIDTH; x++) {
  //   for(int y = 0; y < HEIGHT; y++) {
  //     int i = y * WIDTH + x;
  //     RGB c = Hsv2rgb({(float)data[i], 100, (float)((data[i] % 200) >= 100 ? 100 - (data[i] % 100) : data[i]%100)});
  //     SDL_SetRenderDrawColor(ren, c.r, c.g, c.b, 255);
  //     SDL_RenderDrawPoint(ren, x, y);
  //   }
  // }
  // SDL_RenderPresent(ren);

  SDL_UpdateTexture(texture , NULL, data, WIDTH * sizeof (int));
  SDL_RenderClear(ren);
  SDL_RenderCopy(ren, texture, NULL, NULL);
  SDL_RenderPresent(ren);
}


__global__
void Calculate(int * data, COORD xC,COORD yC, COORD zoom, int iterations, int width, int height) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int k = y * width + x;

  if(x >= width || y >= height)
    return;

  COORD cR = xC + (x - width / 2) / zoom;
  COORD cI = yC + (y - height / 2) / zoom;
  COORD real = (double)0;
  COORD imaginary = (double)0;
  data[k] = 0;
  for(int i = 0; i < iterations; i++) {
    COORD R = real;
    real = R * R - imaginary * imaginary + cR;
    imaginary = 2 * R * imaginary + cI;
    if(abs(real) > 2 || abs(imaginary) > 2) {
      RGB c = Hsv2rgb({(float)i, 100, (float)((i % 200) >= 100 ? 100 - (i % 100) : i%100)});
      data[k] = c.r << 16;
      data[k] += c.g << 8;
      data[k] += c.b;
      break;
    }
  }
}


int main() {
  int imageSize = HEIGHT * WIDTH * sizeof(int);
  int * data;
  int * d_data;
  data = (int *) malloc(imageSize);
  cudaMalloc(&d_data, imageSize);

  for(int x = 0; x < WIDTH; x++)
    for(int y = 0; y < HEIGHT; y++)
      data[y * WIDTH + x] = 255;

  if(SDL_Init(SDL_INIT_VIDEO) != 0) {
    SDL_Log("Unable to initialize SDL: %s", SDL_GetError());
    return 1;
  }

  SDL_Window * win = SDL_CreateWindow("MandelBrot Viewer", 100, 100, WIDTH, HEIGHT, 0);
  SDL_Renderer * ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);
  SDL_Texture * texture = SDL_CreateTexture(ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, WIDTH, HEIGHT);

  int iterations = 1000;
  COORD x = startX;
  COORD y = startY;
  COORD zoom = startZoom;  //150

  dim3 blockSize(WIDTH / 64, HEIGHT / 64);

  int bx = (WIDTH + blockSize.x - 1) / blockSize.x;
  int by = (HEIGHT + blockSize.y - 1) / blockSize.y;

  dim3 gridSize(bx, by);

  Calculate<<<gridSize, blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT);
  cudaMemcpy(data, d_data, imageSize, cudaMemcpyDeviceToHost);
  Draw(ren, texture, data);

  bool printDebug = false;
  bool printPosition = false;

  while(true) {
    SDL_Event event;
    SDL_PollEvent(&event);
    if(event.type == SDL_QUIT)
      break;
    else if(event.type == SDL_KEYDOWN) {
      system_clock::time_point timeStart = system_clock::now();
      switch(event.key.keysym.sym) {
        case SDLK_UP: y -= 1 / zoom * MOVE_AMOUNT;
          break;
        case SDLK_DOWN: y += 1 / zoom * MOVE_AMOUNT;
          break;
        case SDLK_RIGHT: x += 1 / zoom * MOVE_AMOUNT;
          break;
        case SDLK_LEFT: x -= 1 / zoom * MOVE_AMOUNT;
          break;
        case SDLK_PAGEUP: zoom *= 1.1;
          break;
        case SDLK_PAGEDOWN: zoom /= 1.1;
          break;
        case SDLK_KP_PLUS: iterations *= 5;
          std::cout << iterations << std::endl;
          break;
        case SDLK_KP_MINUS: iterations /= 5;
          std::cout << iterations << std::endl;
          break;
        case SDLK_d: printDebug = !printDebug; printPosition = false;
          break;
        case SDLK_p: printPosition = !printPosition; printDebug = false;
          break;
        default: break;
      }
      system_clock::time_point calcStart = system_clock::now();
      Calculate<<<gridSize, blockSize>>>(d_data, x, y, zoom, iterations, WIDTH, HEIGHT);
      duration<double> calcDuration = system_clock::now() - calcStart;

      system_clock::time_point copyStart = system_clock::now();
      cudaMemcpy(data, d_data, imageSize, cudaMemcpyDeviceToHost);
      duration<double> copyDuration = system_clock::now() - copyStart;

      system_clock::time_point drawStart = system_clock::now();
      Draw(ren, texture, data);
      duration<double> drawDuration = system_clock::now() - drawStart;
  
      duration<double> duration = system_clock::now() - timeStart;
      double fps = 1 / duration.count();
      if(printDebug)
        std::cout << std::fixed
        << "FPS: " << fps
        << " calc: " << calcDuration.count()
        << " copy: " << copyDuration.count()
        << " draw: " << drawDuration.count() << std::endl;
      //  if(printPosition)
      //    std::cout << std::setprecision(20) << std::fixed
      //    << "x: " << x 
      //    << " y: " << y << std::setprecision(5) << std::scientific
      //    << " zoom: " << zoom 
      //    << std::endl;

      //std::cout << zoom << std::endl;
    }
  }

  free(data);
  cudaFree(d_data);

  SDL_DestroyRenderer(ren);
  SDL_DestroyWindow(win);
  SDL_Quit();
  return 0;
}
