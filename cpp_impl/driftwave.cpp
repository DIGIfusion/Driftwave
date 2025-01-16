#include "raylib.h"
#include <math.h>
#include <stdio.h>

#include <vector>
using Vector = std::vector<double>;

/* Continuity Equation
Takes
    - arrays:  delta_N
    - scalars: T, N, nref, B, Ln
        - Temperature, Density, Reference density, mag field, density gradient length
Computes:
    - Phi = delta_n*T / (N*nref)
    - d/dy Phi
Returns:

    - dndt = 1/B * (d/dy Phi) * N*nref / Ln
        - Comes from:
            dndt = -v_ExB*grad_n
            grad_n = n*n0/Ln
            v_ExB = (1/B)*-dphidy
    Periodic boundary condition
        - dndt[0] = dndt[N-1]
*/
// Function for solving continuity equation
Vector
continuity_eq(const Vector &delta_N, double T, double N, double nref, double B, double Ln, int NX, double dky)
{
    // float dndt[NX] = {0};
    // float Phi[NX] = {0};
    // float dphidy[NX] = {0};
    Vector Phi(delta_N.size());
    Vector dphidy(delta_N.size());
    Vector dndt(delta_N.size());
    for (size_t i = 0; i < NX; i++)
    {
        Phi[i] = delta_N[i] * T / (N * nref);
    }
    // 4th order central difference
    for (size_t i = 2; i < NX - 2; i++)
    {
        dphidy[i] = (-Phi[i + 2] + 8 * Phi[i + 1] - 8 * Phi[i - 1] + Phi[i - 2]) / (12 * dky);
    }
    // 4th order Central difference assuming periodicity
    dphidy[0] = (-Phi[2] + 8 * Phi[1] - 8 * Phi[NX - 1] + Phi[NX - 2]) / (12 * dky);
    dphidy[1] = (-Phi[3] + 8 * Phi[2] - 8 * Phi[0] + Phi[NX - 1]) / (12 * dky);
    dphidy[NX - 2] = (-Phi[0] + 8 * Phi[NX - 1] - 8 * Phi[NX - 3] + Phi[NX - 4]) / (12 * dky);
    dphidy[NX - 1] = dphidy[0];
    // dphidy[0] = (-Phi[(NX - 2) % NX] + 8 * Phi[(NX - 1) % NX] - 8 * Phi[(1) % NX] + Phi[(2) % NX]) / (12 * dky);
    // dphidy[1] = (-Phi[(NX - 1) % NX] + 8 * Phi[(0) % NX] - 8 * Phi[(2) % NX] + Phi[(3) % NX]) / (12 * dky);
    // dphidy[NX - 2] = (-Phi[(NX - 4) % NX] + 8 * Phi[(NX - 3) % NX] - 8 * Phi[(NX - 1) % NX] + Phi[(0) % NX]) / (12 * dky);
    // dphidy[NX - 1] = dphidy[0]; // Enforce periodicity

    // dphidy[NX - 3] = (Phi[-5] - 8 * Phi[-4] + 8 * Phi[-2] - Phi[0]) / (12 * dky);
    // dphidy[0] = (Phi[-3] - 8 * Phi[-4] + 8 * Phi[-2] - Phi[0]) / (12 * dky);
    // dphidy[1] = (Phi[-2] - 8 * Phi[0] + 8 * Phi[2] - Phi[3]) / (12 * dky);
    // dphidy[NX - 2] = (Phi[-4] - 8 * Phi[-3] + 8 * Phi[0] - Phi[1]) / (12 * dky);
    // dphidy[-1] = dphidy[0];

    for (size_t i = 0; i < NX; i++)
    {
        dndt[i] = (1 / B) * dphidy[i] * N * nref / Ln;
    }
    dndt[0] = dndt[NX - 1]; // Enforce periodicity
    return dndt;
}

// function for timestepping dt forward in time using RK4
Vector forward_dt(Vector delta_N, double T, double N, double nref, double B, double Ln, int NX, double dky, double dt)
{
    Vector temp = delta_N; // copy of delta_N
    Vector k1 = continuity_eq(delta_N, T, N, nref, B, Ln, NX, dky);

    for (size_t i = 0; i < delta_N.size(); i++)
    {
        temp[i] = delta_N[i] + (0.5 * dt) * k1[i];
    }
    Vector k2 = continuity_eq(temp, T, N, nref, B, Ln, NX, dky);

    for (size_t i = 0; i < delta_N.size(); i++)
    {
        temp[i] = delta_N[i] + (0.5 * dt) * k2[i];
    }
    Vector k3 = continuity_eq(temp, T, N, nref, B, Ln, NX, dky);

    for (size_t i = 0; i < delta_N.size(); i++)
    {
        temp[i] = delta_N[i] + dt * k3[i];
    }
    Vector k4 = continuity_eq(temp, T, N, nref, B, Ln, NX, dky);

    for (size_t i = 0; i < delta_N.size(); i++)
    {
        temp[i] = (k1[i] + 2 * k2[2] + 2 * k3[i] + k4[i]) * dt / 6;
    }
    for (size_t i = 0; i < delta_N.size(); i++)
    {
        delta_N[i] = delta_N[i] + temp[i];
    }
    return delta_N;
}

int main(void)
{

    double T = 100;          // Temperature in eV
    double n = 1.0;          // Density in 10^19 m^-3
    double Ln = 0.01;        // Density gradient length in m
    double B = 1.0;          // Magnetic field in T
    double nref = 1.0E19;    // Reference density in m^-3
    static int NX = 500;     // Resolution, within length of 2pi
    static double L_y = 1.0; // domain length in y-direction

    double kygrid[NX] = {0};
    for (int i = 0; i < NX; i++)
    {
        kygrid[i] = 2 * PI * i / (NX * L_y);
    }
    double dky = (kygrid[1] - kygrid[0]) / (2 * PI * L_y);

    Vector density = Vector(NX, 0.0);
    // Initialize perturbation of density
    double ky = 5.0;
    // 0.01*sin(ky*kygrid)*n0
    for (int i = 0; i < NX; i++)
    {
        density[i] = 0.1 * sin(ky * kygrid[i]) * nref;
    }

    // print out the density
    for (int i = 0; i < NX; i++)
    {
        printf("Density[%d] = %f\n", i, density[i] / nref);
    }

    double dt = 1E-7;
    double currtime = 0.0;
    double tlastplot = 0.0;
    int screenWidth = 800;
    int screenHeight = 450;

    InitWindow(screenWidth, screenHeight, "Driftwave");

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        // Update
        density = forward_dt(density, T, n, nref, B, Ln, NX, dky, dt);
        // Draw grid,
        // kygrid on x-axis
        // and y-amplitude is density
        currtime += dt;
        tlastplot += dt;
        // only plot every 0.1 seconds
        BeginDrawing();
        ClearBackground(RAYWHITE);
        for (size_t i = 0; i < NX; i++)
        {
            DrawCircle(i + screenWidth / 4, screenHeight / 2 + 250 * density[i] / nref, 2.0, RED);
        }

        // Additional points
        DrawRectangleLines(screenWidth / 4, screenHeight / 2 - 100, NX, 200, BLACK);
        DrawLine(screenWidth / 4, screenHeight / 2, screenWidth / 4 + NX, screenHeight / 2, GRAY);
        DrawText(TextFormat("Time: %f s", currtime), 10, 30, 20, DARKGRAY);
        DrawText("Driftwave", 10, 10, 20, DARKGRAY);
        DrawText("Density", screenWidth / 2, screenHeight / 2 - 100, 20, DARKBLUE);
        DrawText("ky", screenWidth / 2 + MeasureText("ky", 20), screenHeight / 2 + 100, 20, DARKBLUE);
        DrawText("n/n0", screenWidth / 4 - MeasureText("n / n0", 20) - 5, screenHeight / 2, 20, DARKBLUE);
        EndDrawing();
    }
}