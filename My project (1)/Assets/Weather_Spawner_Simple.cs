using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WeatherSpawnerSimple : MonoBehaviour
{
    [Header("Prefabs")]
    public GameObject ThunderPrefab;
    public GameObject SnowPrefab;
    public GameObject RainPrefab;
    public GameObject SunPrefab;

    [Header("Plane reference")]
    public GameObject MapPlane;

    void Start()
    {
        // Simple example: spawn 4 weather types at different positions
        SpawnWeather(SunPrefab, new Vector3(-5, 0.5f, -5));
        SpawnWeather(SnowPrefab, new Vector3(5, 0.5f, -5));
        SpawnWeather(RainPrefab, new Vector3(-5, 0.5f, 5));
        SpawnWeather(ThunderPrefab, new Vector3(5, 0.5f, 5));
    }

    void SpawnWeather(GameObject prefab, Vector3 pos)
    {
        // Adjust relative to plane
        Vector3 planePos = MapPlane.transform.position;
        Vector3 planeScale = MapPlane.transform.localScale * 10f; // default Unity plane is 10x10
        Vector3 spawnPos = planePos + new Vector3(pos.x, pos.y, pos.z);
        Instantiate(prefab, spawnPos, Quaternion.identity);
    }
}

