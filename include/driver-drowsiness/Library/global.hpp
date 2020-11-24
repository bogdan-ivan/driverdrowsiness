#pragma once
#ifdef Library_EXPORTS
	#define Library_API __declspec(dllexport)
#else
	#define Library_API __declspec(dllimport)
#endif